import json
from functools import lru_cache
from pathlib import Path
from django.conf import settings


@lru_cache(maxsize=1)
def load_directory():
    path = Path(settings.BASE_DIR) / "cases" / "data" / "tamil_nadu_legal_help.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("state") != "Tamil Nadu":
        raise ValueError("The legal-help directory is not configured for Tamil Nadu.")
    return data


def find_tamil_nadu_help(district, specialization=""):
    data = load_directory()
    district_key = district.casefold().strip()
    speciality_key = specialization.casefold().strip()

    districts_list = data.get("districts", [])
    district_info = None

    for d in districts_list:
        if d["district"].casefold() == district_key:
            district_info = d
            break

    if not district_info and district_key:
        for d in districts_list:
            if district_key in d["district"].casefold() or d["district"].casefold() in district_key:
                district_info = d
                break

    lawyers = []
    for item in data.get("demo_lawyers", []):
        if district_key and district_key not in item["district"].casefold():
            continue
        if speciality_key and speciality_key not in item["specializations"]:
            continue
        lawyers.append(item)

    return {
        "district": district,
        "specialization": specialization,
        "district_info": district_info,
        "offices": [district_info["legal_services"]] if district_info and district_info.get("legal_services") else [],
        "courts": [district_info["court_information"]] if district_info and district_info.get("court_information") else [],
        "services": district_info.get("available_services", []) if district_info else [],
        "lawyers": lawyers,
        "directory_updated": data.get("last_updated", "August 2026"),
        "total_districts": data.get("total_districts", 38),
    }


def get_directory_stats():
    data = load_directory()
    districts = data.get("districts", [])
    configured_districts_count = len(districts)
    
    return {
        "configured_districts": configured_districts_count,
        "total_districts": data.get("total_districts", 38),
        "district_websites_count": sum(1 for d in districts if d.get("official_district_website")),
        "dlsa_count": sum(1 for d in districts if d.get("legal_services")),
        "ecourts_count": sum(1 for d in districts if d.get("court_information")),
        "last_updated": data.get("last_updated", "August 2026"),
    }
