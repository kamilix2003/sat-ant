from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math


@dataclass(frozen=True)
class TLE:
	"""Two-Line Element (TLE) record.

	This class stores raw TLE lines and exposes parsed orbital values.
	It supports optional checksum validation and conversion helpers.
	"""

	line1: str
	line2: str
	name: str | None = None

	@classmethod
	def from_orbital_parameters(
		cls,
		*,
		epoch: datetime,
		inclination_deg: float,
		raan_deg: float,
		eccentricity: float,
		argument_of_perigee_deg: float,
		mean_anomaly_deg: float,
		mean_motion_rev_per_day: float,
		name: str | None = None,
		satellite_number: int = 99999,
		classification: str = "U",
		international_designator: str = "00000A",
		mean_motion_dot: float = 0.0,
		mean_motion_ddot: float = 0.0,
		bstar: float = 0.0,
		ephemeris_type: int = 0,
		element_set_number: int = 1,
		revolution_number_at_epoch: int = 1,
		validate: bool = True,
	) -> "TLE":
		"""Build a valid TLE object from epoch and orbital parameters.

		Only orbital parameters and epoch are required; all remaining fields
		can use defaults or be overridden when needed.
		"""
		if not (0.0 <= eccentricity < 1.0):
			raise ValueError("eccentricity must be in [0, 1)")
		if mean_motion_rev_per_day <= 0:
			raise ValueError("mean_motion_rev_per_day must be > 0")

		satnum = f"{satellite_number:05d}"[-5:]
		cls_char = (classification or "U")[0]
		int_desig = (international_designator or "").strip()[:8].ljust(8)

		epoch_yy, epoch_day_str = cls._format_epoch_fields(epoch)
		mm_dot = cls._format_mm_dot(mean_motion_dot)
		mm_ddot = cls._format_tle_exponent(mean_motion_ddot)
		bstar_str = cls._format_tle_exponent(bstar)

		line1_68 = (
			f"1 {satnum}{cls_char} {int_desig} "
			f"{epoch_yy:02d}{epoch_day_str} "
			f"{mm_dot} {mm_ddot} {bstar_str} "
			f"{ephemeris_type:d} {element_set_number:4d}"
		)

		ecc_digits = f"{int(round(eccentricity * 1e7)):07d}"[:7]
		line2_68 = (
			f"2 {satnum} "
			f"{inclination_deg % 360:8.4f} "
			f"{raan_deg % 360:8.4f} "
			f"{ecc_digits} "
			f"{argument_of_perigee_deg % 360:8.4f} "
			f"{mean_anomaly_deg % 360:8.4f} "
			f"{mean_motion_rev_per_day:11.8f}"
			f"{revolution_number_at_epoch:5d}"
		)

		if len(line1_68) != 68:
			raise ValueError(f"Internal formatting error: line1 length is {len(line1_68)}, expected 68")
		if len(line2_68) != 68:
			raise ValueError(f"Internal formatting error: line2 length is {len(line2_68)}, expected 68")

		line1 = f"{line1_68}{cls._checksum(line1_68)}"
		line2 = f"{line2_68}{cls._checksum(line2_68)}"

		return cls.from_lines(line1=line1, line2=line2, name=name, validate=validate)

	@classmethod
	def from_orbital_elements(cls, **kwargs) -> "TLE":
		"""Alias for from_orbital_parameters."""
		return cls.from_orbital_parameters(**kwargs)

	@classmethod
	def from_lines(
		cls,
		line1: str,
		line2: str,
		name: str | None = None,
		validate: bool = True,
	) -> "TLE":
		"""Build a TLE object from two lines (and optional satellite name)."""
		line1 = line1.strip()
		line2 = line2.strip()
		name = name.strip() if name is not None else None

		tle = cls(line1=line1, line2=line2, name=name)
		if validate:
			tle.validate()
		return tle

	@classmethod
	def from_string(cls, text: str, validate: bool = True) -> "TLE":
		"""Parse TLE from a 2-line or 3-line string.

		Accepted formats:
		1. line1 + line2
		2. name + line1 + line2
		"""
		rows = [line.strip() for line in text.splitlines() if line.strip()]
		if len(rows) == 2:
			name = None
			line1, line2 = rows
		elif len(rows) == 3:
			name, line1, line2 = rows
		else:
			raise ValueError("TLE text must contain exactly 2 or 3 non-empty lines")

		return cls.from_lines(line1=line1, line2=line2, name=name, validate=validate)

	def validate(self) -> None:
		"""Validate line prefixes, length, and checksums."""
		if not self.line1.startswith("1 "):
			raise ValueError("line1 must start with '1 '")
		if not self.line2.startswith("2 "):
			raise ValueError("line2 must start with '2 '")

		if len(self.line1) < 69:
			raise ValueError("line1 must be at least 69 characters")
		if len(self.line2) < 69:
			raise ValueError("line2 must be at least 69 characters")

		sat1 = self.line1[2:7]
		sat2 = self.line2[2:7]
		if sat1 != sat2:
			raise ValueError("line1 and line2 satellite numbers do not match")

		c1 = self._checksum(self.line1[:68])
		c2 = self._checksum(self.line2[:68])
		if self.line1[68].isdigit() and int(self.line1[68]) != c1:
			raise ValueError("line1 checksum mismatch")
		if self.line2[68].isdigit() and int(self.line2[68]) != c2:
			raise ValueError("line2 checksum mismatch")

	@staticmethod
	def _format_epoch_fields(epoch: datetime) -> tuple[int, str]:
		"""Return (2-digit year, day-of-year string) for TLE epoch fields."""
		if epoch.tzinfo is None:
			epoch = epoch.replace(tzinfo=timezone.utc)
		else:
			epoch = epoch.astimezone(timezone.utc)

		year2 = epoch.year % 100
		start = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
		day_of_year = 1 + (epoch - start).total_seconds() / 86400.0
		return year2, f"{day_of_year:012.8f}"

	@staticmethod
	def _format_mm_dot(value: float) -> str:
		"""Format mean motion first derivative for TLE line 1 (10 chars)."""
		frac = f"{abs(value):.8f}"
		if frac.startswith("0"):
			frac = frac[1:]
		sign = "-" if value < 0 else " "
		out = f"{sign}{frac}"
		if len(out) != 10:
			raise ValueError("mean_motion_dot cannot be represented in TLE field")
		return out

	@staticmethod
	def _format_tle_exponent(value: float) -> str:
		"""Format compact TLE exponent field (8 chars), e.g. ' 29639-3'."""
		if value == 0:
			return " 00000+0"

		sign = "-" if value < 0 else " "
		v = abs(value)
		e = math.floor(math.log10(v))
		exponent = e + 1
		mantissa = v / (10 ** exponent)
		mantissa_int = int(round(mantissa * 1e5))

		if mantissa_int >= 100000:
			mantissa_int = 10000
			exponent += 1

		if abs(exponent) > 9:
			raise ValueError("value exponent out of range for TLE compact field")

		exp_sign = "+" if exponent >= 0 else "-"
		out = f"{sign}{mantissa_int:05d}{exp_sign}{abs(exponent):1d}"
		if len(out) != 8:
			raise ValueError("Internal formatting error for compact exponent")
		return out

	@staticmethod
	def _checksum(line68: str) -> int:
		"""Compute standard TLE checksum for the first 68 characters."""
		total = 0
		for ch in line68:
			if ch.isdigit():
				total += int(ch)
			elif ch == "-":
				total += 1
		return total % 10

	@staticmethod
	def _parse_tle_exponent(value: str) -> float:
		"""Parse TLE compact scientific notation, e.g. ' 29639-3'."""
		value = value.strip()
		if not value:
			return 0.0

		sign = -1 if value[0] == "-" else 1
		mantissa_digits = value[1:6] if value[0] in "+-" else value[0:5]
		exponent = int(value[-2:])
		mantissa = int(mantissa_digits) / 1e5
		return sign * mantissa * (10 ** exponent)

	@staticmethod
	def _parse_epoch(year2: int, day_of_year: float) -> datetime:
		"""Convert TLE epoch year/day fields into UTC datetime."""
		year = 2000 + year2 if year2 < 57 else 1900 + year2
		start = datetime(year, 1, 1, tzinfo=timezone.utc)
		return start + timedelta(days=day_of_year - 1)

	@property
	def satellite_number(self) -> int:
		return int(self.line1[2:7])

	@property
	def classification(self) -> str:
		return self.line1[7]

	@property
	def international_designator(self) -> str:
		return self.line1[9:17].strip()

	@property
	def epoch_year_2digit(self) -> int:
		return int(self.line1[18:20])

	@property
	def epoch_day(self) -> float:
		return float(self.line1[20:32])

	@property
	def epoch_datetime(self) -> datetime:
		return self._parse_epoch(self.epoch_year_2digit, self.epoch_day)

	@property
	def mean_motion_dot(self) -> float:
		return float(self.line1[33:43])

	@property
	def mean_motion_ddot(self) -> float:
		return self._parse_tle_exponent(self.line1[44:52])

	@property
	def bstar(self) -> float:
		return self._parse_tle_exponent(self.line1[53:61])

	@property
	def ephemeris_type(self) -> int:
		return int(self.line1[62])

	@property
	def element_set_number(self) -> int:
		return int(self.line1[64:68])

	@property
	def inclination_deg(self) -> float:
		return float(self.line2[8:16])

	@property
	def raan_deg(self) -> float:
		return float(self.line2[17:25])

	@property
	def eccentricity(self) -> float:
		return float(f"0.{self.line2[26:33].strip()}")

	@property
	def argument_of_perigee_deg(self) -> float:
		return float(self.line2[34:42])

	@property
	def mean_anomaly_deg(self) -> float:
		return float(self.line2[43:51])

	@property
	def mean_motion_rev_per_day(self) -> float:
		return float(self.line2[52:63])

	@property
	def revolution_number_at_epoch(self) -> int:
		return int(self.line2[63:68])

	def to_dict(self) -> dict:
		"""Return raw lines and parsed orbital elements as a dictionary."""
		parsed = {
			"satellite_number": self.satellite_number,
			"classification": self.classification,
			"international_designator": self.international_designator,
			"epoch_year_2digit": self.epoch_year_2digit,
			"epoch_day": self.epoch_day,
			"epoch_datetime": self.epoch_datetime.isoformat(),
			"mean_motion_dot": self.mean_motion_dot,
			"mean_motion_ddot": self.mean_motion_ddot,
			"bstar": self.bstar,
			"ephemeris_type": self.ephemeris_type,
			"element_set_number": self.element_set_number,
			"inclination_deg": self.inclination_deg,
			"raan_deg": self.raan_deg,
			"eccentricity": self.eccentricity,
			"argument_of_perigee_deg": self.argument_of_perigee_deg,
			"mean_anomaly_deg": self.mean_anomaly_deg,
			"mean_motion_rev_per_day": self.mean_motion_rev_per_day,
			"revolution_number_at_epoch": self.revolution_number_at_epoch,
		}

		return {
			**asdict(self),
			**parsed,
		}

	def __str__(self) -> str:
		if self.name:
			return f"{self.name}\n{self.line1}\n{self.line2}"
		return f"{self.line1}\n{self.line2}"

