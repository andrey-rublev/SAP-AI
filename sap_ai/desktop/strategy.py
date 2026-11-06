"""Named desktop strategies shared by offline studies and the launcher."""
from sap_ai.desktop.state import DesktopPolicy


STRATEGIES = ("conservative", "economy")


def make_policy(name="conservative", *, sale_funded_rolls=None):
    """Build a fresh policy; the economy variant fills the team before merging.

    A funded roll authorizes only the refresh. Any sale and purchase still need
    a fresh observed board and their own acknowledged receipts.
    """
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")
    if sale_funded_rolls is not None and type(sale_funded_rolls) is not bool:
        raise ValueError("sale_funded_rolls must be boolean or None")
    return DesktopPolicy(merge_first_team_size=5 if name == "economy" else 3,
                         sale_funded_rolls=(name == "economy" if sale_funded_rolls is None else sale_funded_rolls))
