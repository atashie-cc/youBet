"""Team name normalization to match between Kaggle, KenPom, CBB, and Elo datasets.

Each dataset uses slightly different naming conventions. This module provides
a normalization function that maps all names to a canonical form.
"""

from __future__ import annotations

import re


# Explicit mappings for names that can't be resolved by rules alone.
# Format: kaggle_name_lower -> kenpom_name_lower
KAGGLE_TO_KENPOM = {
    "abilene chr": "abilene christian",
    "alabama a&m": "alabama a&m",
    "alabama st": "alabama state",
    "albany ny": "albany",
    "alcorn st": "alcorn state",
    "alliant intl": "alliant",
    "american univ": "american",
    "appalachian st": "app state",
    "arizona st": "arizona state",
    "ark little rock": "little rock",
    "ark pine bluff": "arkansas-pine bluff",
    "arkansas st": "arkansas state",
    "armstrong st": "armstrong",
    "ball st": "ball state",
    "birmingham so": "birmingham-southern",
    "boise st": "boise state",
    "boston univ": "boston university",
    "brooklyn": "liu brooklyn",
    "c michigan": "central michigan",
    "cal poly slo": "cal poly",
    "cal st bakersfield": "cal state bakersfield",
    "cal st fullerton": "cal state fullerton",
    "cal st northridge": "cal state northridge",
    "central conn": "central connecticut",
    "chas southern": "charleston southern",
    "chicago st": "chicago state",
    "cleveland st": "cleveland state",
    "col charleston": "charleston",
    "colorado st": "colorado state",
    "coppin st": "coppin state",
    "cs sacramento": "sacramento state",
    "delaware st": "delaware state",
    "e illinois": "eastern illinois",
    "e kentucky": "eastern kentucky",
    "e michigan": "eastern michigan",
    "e washington": "eastern washington",
    "f dickinson": "fairleigh dickinson",
    "fiu": "fiu",
    "fl atlantic": "florida atlantic",
    "florida a&m": "florida a&m",
    "florida intl": "fiu",
    "florida st": "florida state",
    "fort wayne": "purdue fort wayne",
    "fresno st": "fresno state",
    "ga southern": "georgia southern",
    "gardner webb": "gardner-webb",
    "geo washington": "george washington",
    "georgia st": "georgia state",
    "georgia tech": "georgia tech",
    "grambling": "grambling state",
    "green bay": "green bay",
    "houston bap": "houston christian",
    "idaho st": "idaho state",
    "il chicago": "uic",
    "illinois st": "illinois state",
    "in ft wayne": "purdue fort wayne",
    "indiana st": "indiana state",
    "iowa st": "iowa state",
    "jackson st": "jackson state",
    "jackson ville": "jacksonville",
    "jksonville st": "jacksonville state",
    "kansas st": "kansas state",
    "kennesaw": "kennesaw state",
    "kent": "kent state",
    "lamar univ": "lamar",
    "long beach st": "long beach state",
    "long island": "liu",
    "loyola il": "loyola chicago",
    "loyola md": "loyola maryland",
    "loyola mymt": "loyola marymount",
    "lsu": "lsu",
    "md baltimore county": "umbc",
    "md east shore": "maryland-eastern shore",
    "memphis st": "memphis",
    "miami fl": "miami",
    "miami oh": "miami (oh)",
    "michigan st": "michigan state",
    "mid tenn st": "middle tennessee",
    "miss valley st": "mississippi valley state",
    "mississippi": "ole miss",
    "mississippi st": "mississippi state",
    "montana st": "montana state",
    "morehead st": "morehead state",
    "morgan st": "morgan state",
    "mt st mary's": "mount st. mary's",
    "murray st": "murray state",
    "n carolina a&t": "north carolina a&t",
    "n carolina st": "nc state",
    "n colorado": "northern colorado",
    "n dakota st": "north dakota state",
    "n illinois": "northern illinois",
    "n iowa": "northern iowa",
    "n kentucky": "northern kentucky",
    "n mex state": "new mexico state",
    "nc asheville": "unc asheville",
    "nc central": "north carolina central",
    "nc greensboro": "unc greensboro",
    "nc wilmington": "unc wilmington",
    "new mexico st": "new mexico state",
    "nicholls st": "nicholls state",
    "norfolk st": "norfolk state",
    "oak state": "oakland",
    "ohio st": "ohio state",
    "oklahoma st": "oklahoma state",
    "old dominion": "old dominion",
    "oral roberts": "oral roberts",
    "oregon st": "oregon state",
    "penn st": "penn state",
    "portland st": "portland state",
    "prairie view": "prairie view a&m",
    "sacred heart": "sacred heart",
    "sam houston st": "sam houston state",
    "san diego st": "san diego state",
    "san jose st": "san jose state",
    "savannah st": "savannah state",
    "se louisiana": "southeastern louisiana",
    "se missouri st": "southeast missouri state",
    "siu edwardsville": "siu edwardsville",
    "south carolina st": "south carolina state",
    "south dakota st": "south dakota state",
    "southern ill": "southern illinois",
    "southern miss": "southern miss",
    "southern univ": "southern",
    "st bonaventure": "st. bonaventure",
    "st francis ny": "st. francis (ny)",
    "st francis pa": "st. francis (pa)",
    "st john's": "st. john's",
    "st joseph's": "saint joseph's",
    "st joseph pa": "saint joseph's",
    "st louis": "saint louis",
    "st mary's": "saint mary's",
    "st peter's": "saint peter's",
    "stephen f austin": "stephen f. austin",
    "sw missouri st": "missouri state",
    "tenn martin": "ut martin",
    "tenn st": "tennessee state",
    "tenn tech": "tennessee tech",
    "texas a&m cc": "texas a&m-corpus christi",
    "texas pan am": "ut rio grande valley",
    "texas so": "texas southern",
    "texas st": "texas state",
    "uc davis": "uc davis",
    "uc irvine": "uc irvine",
    "uc riverside": "uc riverside",
    "uc santa barb": "uc santa barbara",
    "ucf": "ucf",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "umass lowell": "umass lowell",
    "umkc": "umkc",
    "unlv": "unlv",
    "usc": "usc",
    "ut arlington": "ut arlington",
    "ut chattanooga": "chattanooga",
    "ut san antonio": "utsa",
    "utah st": "utah state",
    "utah valley": "utah valley",
    "utep": "utep",
    "va commonwealth": "vcu",
    "virginia tech": "virginia tech",
    "w carolina": "western carolina",
    "w illinois": "western illinois",
    "w kentucky": "western kentucky",
    "w michigan": "western michigan",
    "w virginia": "west virginia",
    "wash state": "washington state",
    "weber st": "weber state",
    "wichita st": "wichita state",
    "wm & mary": "william & mary",
    "wright st": "wright state",
    "youngstown st": "youngstown state",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to a canonical form for matching across datasets.

    Applies rule-based normalization, then checks explicit mappings.
    """
    if not isinstance(name, str):
        return ""

    lower = name.lower().strip()

    # Check explicit mapping first
    if lower in KAGGLE_TO_KENPOM:
        return KAGGLE_TO_KENPOM[lower]

    # Rule-based normalization
    normalized = lower

    # Remove common suffixes/prefixes that vary
    normalized = re.sub(r'\s+', ' ', normalized)

    # Common abbreviation expansions
    replacements = [
        (r'\bst\b$', 'state'),
        (r'\bst\b(?!\.*\s)', 'st.'),  # "St" at start -> "St."
        (r'\buniv\b', 'university'),
        (r'\bchr\b', 'christian'),
    ]

    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)

    return normalized.strip()


def build_name_lookup(kaggle_names: list[str], kenpom_names: list[str]) -> dict[str, str]:
    """Build a lookup from any team name variant to its canonical (KenPom) form."""
    lookup = {}

    # KenPom names are canonical
    for name in kenpom_names:
        lower = name.lower().strip()
        lookup[lower] = lower

    # Map Kaggle names to KenPom names
    for name in kaggle_names:
        lower = name.lower().strip()
        normalized = normalize_team_name(name)
        if normalized in lookup:
            lookup[lower] = normalized
        elif lower not in lookup:
            # Try to find a close match in KenPom names
            lookup[lower] = normalized  # Best effort

    return lookup
