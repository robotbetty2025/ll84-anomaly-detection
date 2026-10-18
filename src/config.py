"""Project configuration for the LL84 building-energy anomaly detection capstone (DTSC 691)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_DIR = ROOT.parent
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

RANDOM_STATE = 42
TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024

# Plausibility limits (from the approved proposal)
EUI_MIN, EUI_MAX = 5, 1000          # kBtu/ft2
GFA_MIN = 25_000                     # ft2 (LL84 covers buildings > 25k sf)
YEAR_BUILT_MIN = 1800
MIN_BUILDINGS_PER_TYPE = 50
MAX_MISSING_FRAC = 0.40              # drop features missing more than this
EXCLUDED_TYPES = {"Parking"}         # EUI per ft2 is not meaningful for parking structures


def find_raw_csv() -> Path:
    """Locate the raw LL84 CSV. Override with the LL84_CSV environment variable."""
    env = os.environ.get("LL84_CSV")
    if env:
        return Path(env)
    for base in (ROOT / "data" / "raw", CAPSTONE_DIR):
        hits = sorted(base.glob("NYC_Building_Energy*.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError("LL84 CSV not found. Put it in data/raw/ or set LL84_CSV.")


# raw column name -> short name
RAW_COLS = {
    "Calendar Year": "year",
    "Property ID": "property_id",
    "Property Name": "property_name",
    "NYC Borough, Block and Lot (BBL)": "bbl",
    "Address 1": "address",
    "Borough": "borough",
    "Primary Property Type - Portfolio Manager-Calculated": "prop_type",
    "National Median Reference Property Type": "ref_type",
    "Largest Property Use Type - Gross Floor Area (ft²)": "gfa_largest_use",
    "Year Built": "year_built",
    "Construction Status": "construction_status",
    "Number of Buildings": "n_buildings",
    "Occupancy": "occupancy",
    "ENERGY STAR Score": "es_score",
    "Site EUI (kBtu/ft²)": "site_eui",
    "Weather Normalized Site EUI (kBtu/ft²)": "wn_site_eui",
    "National Median Site EUI (kBtu/ft²)": "nat_median_site_eui",
    "Source EUI (kBtu/ft²)": "source_eui",
    "Site Energy Use (kBtu)": "site_energy_kbtu",
    "Electricity Use - Grid Purchase (kBtu)": "e_grid_kbtu",
    "Electricity Use – Generated from Onsite Renewable Systems and Used Onsite (kBtu)": "e_onsite_kbtu",
    "Natural Gas Use (kBtu)": "gas_kbtu",
    "District Steam Use (kBtu)": "steam_kbtu",
    "District Hot Water Use (kBtu)": "dhw_kbtu",
    "District Chilled Water Use (kBtu)": "dchw_kbtu",
    "Fuel Oil #1 Use (kBtu)": "oil1_kbtu",
    "Fuel Oil #2 Use (kBtu)": "oil2_kbtu",
    "Fuel Oil #4 Use (kBtu)": "oil4_kbtu",
    "Fuel Oil #5 & 6 Use (kBtu)": "oil56_kbtu",
    "Diesel #2 Use (kBtu)": "diesel_kbtu",
    "Propane Use (kBtu)": "propane_kbtu",
    "Kerosene Use (kBtu)": "kerosene_kbtu",
    "Office - Weekly Operating Hours": "hrs_office",
    "Medical Office - Weekly Operating Hours": "hrs_medoffice",
    "Financial Office - Weekly Operating Hours": "hrs_finoffice",
    "Restaurant - Weekly Operating Hours": "hrs_restaurant",
    "Default Values": "default_values",
    "Estimated Values - Energy": "estimated_energy",
    "Alert - Energy Meter has gaps": "alert_gaps",
    "Report Submission Date": "report_submitted",
    "Latitude": "lat",
    "Longitude": "lon",
}

NUMERIC_COLS = [
    "gfa_largest_use", "year_built", "n_buildings", "occupancy", "es_score", "site_eui",
    "wn_site_eui", "nat_median_site_eui", "source_eui", "site_energy_kbtu", "e_grid_kbtu",
    "e_onsite_kbtu", "gas_kbtu", "steam_kbtu", "dhw_kbtu", "dchw_kbtu", "oil1_kbtu", "oil2_kbtu",
    "oil4_kbtu", "oil56_kbtu", "diesel_kbtu", "propane_kbtu", "kerosene_kbtu", "hrs_office",
    "hrs_medoffice", "hrs_finoffice", "hrs_restaurant", "lat", "lon",
]
