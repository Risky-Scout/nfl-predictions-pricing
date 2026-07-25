from __future__ import annotations

import re
from typing import Final

# Internal convention follows current nflverse aliases. Historical franchise
# identities remain stable through relocations.
_CANONICAL: Final[dict[str, str]] = {
    # Current aliases
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC",
    "LV": "LV", "LAC": "LAC", "LAR": "LAR", "MIA": "MIA",
    "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF": "SF", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
    # Provider/historical aliases
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "LVR": "LV",
    "OAK": "LV", "SD": "LAC", "SDG": "LAC", "LA": "LAR",
    "STL": "LAR", "NWE": "NE", "NOR": "NO", "SFO": "SF",
    "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
    # Houston Oilers is disambiguated by full team name before alias mapping.
    "HOUSTON OILERS": "TEN",
    "TENNESSEE OILERS": "TEN",
    # Full names
    "ARIZONA CARDINALS": "ARI",
    "PHOENIX CARDINALS": "ARI",
    "ST LOUIS CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "BALTIMORE COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "OAKLAND RAIDERS": "LV",
    "LOS ANGELES RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "SAN DIEGO CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "ST LOUIS RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "BOSTON PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SEATTLE SEAHAWKS": "SEA",
    "SAN FRANCISCO 49ERS": "SF",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "WASHINGTON REDSKINS": "WAS",
}

_PICK_TOKENS = {"PICK", "PK", "EVEN", "PICKEM", "PICK'EM"}


def _normalize(value: str) -> str:
    value = str(value).strip().upper()
    value = value.replace("&", " AND ")
    value = re.sub(r"[^A-Z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_team_id(value: str | None, *, allow_pick: bool = False) -> str | None:
    if value is None:
        return None
    normalized = _normalize(value)
    if not normalized:
        return None
    if normalized in _PICK_TOKENS:
        if allow_pick:
            return "PICK"
        raise ValueError(f"{value!r} is a market pick'em token, not a team.")
    if normalized not in _CANONICAL:
        raise ValueError(f"Unknown NFL team identifier: {value!r}")
    return _CANONICAL[normalized]


def try_canonical_team_id(value: str | None, *, allow_pick: bool = False) -> str | None:
    try:
        return canonical_team_id(value, allow_pick=allow_pick)
    except ValueError:
        return None
