import os
import re
import xml.etree.ElementTree as ET

from models.tree import TreeNode


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MINIMUM_VIABLE_FIELD_COUNT = 5

# Maps known field name aliases to their canonical names.
# Applied before whitelist filtering so aliases always resolve correctly.
# Based on 49-country field analysis. Revisit if new aliases surface.
FIELD_NAME_MAP = {
    "water_percent": "percent_water",
    "pop_est": "population_estimate",
    "population": "population_estimate",
    "area": "area_km2",
    "total_area_km2": "area_km2",
    "official_language": "official_languages",
    "languages_official": "official_languages",
    "calling_codes": "calling_code",
    "timezone_offset": "utc_offset",
    "ethnicity": "ethnic_groups",
    "ethnicities": "ethnic_groups",
}

# Only fields in this set enter the tree. Everything else is silently dropped.
# Decisions grounded in 49-country field analysis (field_analysis.json).
# Frequency threshold: >= 75%. Lower-frequency fields included only where
# semantically important (ethnic_groups, religion, official_languages).
WHITELIST = {
    # Identity (98-100% frequency, universally present)
    "conventional_long_name", "common_name", "capital", "largest_city", "demonym",
    "official_languages", "regional_languages", "languages_type",

    # Geography (92-96% frequency)
    "area_km2", "area_sq_mi", "area_rank", "percent_water",

    # Population (77-91% frequency)
    "population_estimate", "population_estimate_year", "population_estimate_rank",
    "population_census", "population_census_year", "population_census_rank",
    "population_density_km2", "population_density_sq_mi", "population_density_rank",
    "population_rank",

    # Government (82-98% frequency)
    "government_type", "sovereignty_type", "legislature", "upper_house", "lower_house",
    "leader_title1", "leader_name1",
    "leader_title2", "leader_name2",
    "leader_title3", "leader_name3",
    "leader_title4", "leader_name4",
    "established_event1", "established_date1",
    "established_event2", "established_date2",
    "established_event3", "established_date3",
    "established_event4", "established_date4",
    "established_event5", "established_date5",
    "established_event6", "established_date6",
    "established_event7", "established_date7",

    # Economy (92-98% frequency)
    "gdp_ppp", "gdp_ppp_rank", "gdp_ppp_year",
    "gdp_ppp_per_capita", "gdp_ppp_per_capita_rank",
    "gdp_nominal", "gdp_nominal_rank", "gdp_nominal_year",
    "gdp_nominal_per_capita", "gdp_nominal_per_capita_rank",
    "gini", "gini_year", "gini_rank", "gini_change",
    "hdi", "hdi_year", "hdi_rank", "hdi_change",
    "currency", "currency_code",

    # Society (included despite lower frequency, semantically important)
    "religion", "ethnic_groups",
    "drives_on", "calling_code",
    "utc_offset", "utc_offset_dst", "time_zone",
}

# Content nodes under these fields are tagged non_comparable = True.
# Their structural parent participates normally in TED.
# The content value does not affect the similarity score.
# Patching is unaffected; nodes remain in the tree and edit script.
#
# Rationale per field (from sample inspection):
#   conventional_long_name / common_name: country name variants, not comparable
#   capital / largest_city: city names are country identifiers
#   currency / currency_code: assigned identifiers, not meaningful quantities
#   leader_name1..4: person names, not comparable roles
#   calling_code: assigned numeric identifier
#   utc_offset / utc_offset_dst / time_zone: assigned identifiers
NON_COMPARABLE_FIELDS = {
    "conventional_long_name", "common_name",
    "capital", "largest_city",
    "currency", "currency_code",
    "leader_name1", "leader_name2", "leader_name3", "leader_name4",
    "calling_code",
    "utc_offset", "utc_offset_dst", "time_zone",
}

# Fields stored as a single atomic content node (no tokenization)
ATOMIC_FIELDS = {
    "government_type", "demonym", "sovereignty_type",
    "languages_type", "conventional_long_name", "common_name",
    "legislature", "upper_house", "lower_house",
    "currency", "currency_code", "calling_code",
    "time_zone", "drives_on",
    "capital", "largest_city",
}

# Fields parsed into percentage-based subtrees, sorted alphabetically
LIST_FIELDS = {
    "religion", "ethnic_groups",
}

# Fields parsed as comma-separated language entries
LANGUAGE_FIELDS = {
    "official_languages", "regional_languages",
    "languages", "languages2",
}

# Numeric fields stored as a single cleaned numeric value
NUMERIC_FIELDS = {
    "area_km2", "area_sq_mi", "area_rank", "percent_water",
    "population_estimate", "population_census",
    "population_density_km2", "population_density_sq_mi",
    "population_rank", "population_estimate_rank",
    "population_census_rank", "population_density_rank",
    "gdp_ppp", "gdp_ppp_rank", "gdp_ppp_year",
    "gdp_ppp_per_capita", "gdp_ppp_per_capita_rank",
    "gdp_nominal", "gdp_nominal_rank", "gdp_nominal_year",
    "gdp_nominal_per_capita", "gdp_nominal_per_capita_rank",
    "gini", "gini_year", "gini_rank",
    "hdi", "hdi_year", "hdi_rank",
    "utc_offset", "utc_offset_dst",
}

# Unit suffix multipliers for numeric normalization.
# Handles values like "$78.233 billion" -> "78233000000"
UNIT_MULTIPLIERS = {
    "trillion": 1_000_000_000_000,
    "billion":  1_000_000_000,
    "million":  1_000_000,
    "thousand": 1_000,
}


def _normalize_field_name(tag: str) -> str:
    return FIELD_NAME_MAP.get(tag, tag)


def _is_whitelisted(tag: str) -> bool:
    return tag in WHITELIST


def _is_structural_root(tag: str) -> bool:
    # Root and name elements must pass through even though they are not on the whitelist
    return tag in {"country", "name"}


def _clean_numeric(text: str) -> str:
    cleaned = re.sub(r"\[.*?\]", "", text.strip())
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    lower = cleaned.lower()
    for unit, multiplier in UNIT_MULTIPLIERS.items():
        if unit in lower:
            number_part = re.sub(r"[^\d.]", "", lower.split(unit)[0])
            if number_part:
                try:
                    value = float(number_part) * multiplier
                    return str(int(value)) if value == int(value) else str(value)
                except ValueError:
                    pass

    cleaned = re.sub(r"[^\d.\-]", "", cleaned)
    return cleaned or text.strip()


def _tokenize(text: str) -> list[str]:
    tokens = []
    current = []

    for char in text:
        if char in (" ", "\t", "\n", ",", ";", "!", "?", ":", "/"):
            if current:
                tokens.append("".join(current))
                current = []
        elif char == "." and current and not any(c.isdigit() for c in current):
            if current:
                tokens.append("".join(current))
                current = []
        elif char.isupper() and current and current[-1].islower():
            tokens.append("".join(current))
            current = [char]
        else:
            current.append(char)

    if current:
        tokens.append("".join(current))

    return [t for t in tokens if t]


def _make_content_node(label: str, field_name: str) -> TreeNode:
    node = TreeNode(label=label, is_content=True)
    if field_name in NON_COMPARABLE_FIELDS:
        node.non_comparable = True
    return node


def _build_tree(element: ET.Element) -> TreeNode | None:
    tag = _normalize_field_name(element.tag)

    if not _is_whitelisted(tag) and not _is_structural_root(tag):
        return None

    node = TreeNode(label=tag, is_content=False)

    for attr_name in sorted(element.attrib.keys()):
        normalized_attr = _normalize_field_name(attr_name)
        if not _is_whitelisted(normalized_attr):
            continue
        attr_node = TreeNode(label=normalized_attr, is_content=False)
        attr_node.add_child(_make_content_node(element.attrib[attr_name], normalized_attr))
        node.add_child(attr_node)

    if element.text and element.text.strip():
        text = element.text.strip()
        if tag in NUMERIC_FIELDS:
            node.add_child(_make_content_node(_clean_numeric(text), tag))
        elif tag in ATOMIC_FIELDS or tag in LIST_FIELDS or tag in LANGUAGE_FIELDS:
            node.add_child(_make_content_node(text, tag))
        else:
            for token in _tokenize(text):
                node.add_child(_make_content_node(token, tag))

    for child_element in element:
        child_node = _build_tree(child_element)
        if child_node is not None:
            node.add_child(child_node)

    return node


def _find_child(root: TreeNode, label: str) -> TreeNode | None:
    for child in root.children:
        if child.label == label:
            return child
    return None


def _detach(root: TreeNode, node: TreeNode) -> None:
    if node in root.children:
        root.children.remove(node)
        node.parent = None


def _attach(root: TreeNode, node: TreeNode) -> None:
    node.parent = root
    root.children.append(node)


def _make_node(label: str, children_from: TreeNode | None = None) -> TreeNode:
    node = TreeNode(label=label, is_content=False)
    if children_from is not None:
        for child in children_from.children:
            child.parent = node
        node.children = list(children_from.children)
    return node


def _collect_numbered(root: TreeNode, prefix: str) -> dict[int, TreeNode]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    result: dict[int, TreeNode] = {}
    for child in root.children:
        match = pattern.match(child.label)
        if match:
            result[int(match.group(1))] = child
    return result


def _group_population(root: TreeNode) -> None:
    fields = {
        "estimate":       "population_estimate",
        "estimate_year":  "population_estimate_year",
        "estimate_rank":  "population_estimate_rank",
        "census":         "population_census",
        "census_year":    "population_census_year",
        "census_rank":    "population_census_rank",
        "density_km2":    "population_density_km2",
        "density_sq_mi":  "population_density_sq_mi",
        "density_rank":   "population_density_rank",
        "rank":           "population_rank",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    population = TreeNode(label="population", is_content=False)
    for key, node in found.items():
        if node is not None:
            population.add_child(_make_node(key, node))

    _attach(root, population)


def _group_gdp(root: TreeNode, prefix: str, group_label: str) -> None:
    fields = {
        "total":          prefix,
        "rank":           f"{prefix}_rank",
        "year":           f"{prefix}_year",
        "per_capita":     f"{prefix}_per_capita",
        "per_capita_rank": f"{prefix}_per_capita_rank",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    gdp_node = TreeNode(label=group_label, is_content=False)
    for key, node in found.items():
        if node is not None:
            gdp_node.add_child(_make_node(key, node))

    _attach(root, gdp_node)


def _group_index(root: TreeNode, prefix: str, group_label: str) -> None:
    fields = {
        "value":  prefix,
        "year":   f"{prefix}_year",
        "rank":   f"{prefix}_rank",
        "change": f"{prefix}_change",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    index_node = TreeNode(label=group_label, is_content=False)
    for key, node in found.items():
        if node is not None:
            index_node.add_child(_make_node(key, node))

    _attach(root, index_node)


def _group_leaders(root: TreeNode) -> None:
    titles = _collect_numbered(root, "leader_title")
    names = _collect_numbered(root, "leader_name")

    all_numbers = sorted(set(titles.keys()) | set(names.keys()))
    if not all_numbers:
        return

    for node in list(titles.values()) + list(names.values()):
        _detach(root, node)

    leaders = TreeNode(label="leaders", is_content=False)

    for n in all_numbers:
        leader_node = TreeNode(label=f"leader{n}", is_content=False)

        if n in titles:
            title_node = _make_node("title")
            title_val = " ".join(c.label for c in titles[n].children if c.is_content)
            title_node.add_child(TreeNode(label=title_val, is_content=True))
            leader_node.add_child(title_node)

        if n in names:
            name_node = _make_node("name")
            name_val = " ".join(c.label for c in names[n].children if c.is_content)
            name_content = TreeNode(label=name_val, is_content=True)
            name_content.non_comparable = True
            name_node.add_child(name_content)
            leader_node.add_child(name_node)

        leaders.add_child(leader_node)

    _attach(root, leaders)


def _group_establishment(root: TreeNode) -> None:
    events = _collect_numbered(root, "established_event")
    dates = _collect_numbered(root, "established_date")

    all_numbers = sorted(set(events.keys()) | set(dates.keys()))
    if not all_numbers:
        return

    for node in list(events.values()) + list(dates.values()):
        _detach(root, node)

    establishment = TreeNode(label="establishment", is_content=False)

    for n in all_numbers:
        event_node = TreeNode(label=f"event{n}", is_content=False)

        if n in events:
            event_val = " ".join(c.label for c in events[n].children if c.is_content)
            event_node.add_child(TreeNode(label=event_val, is_content=True))

        if n in dates:
            date_val = " ".join(c.label for c in dates[n].children if c.is_content)
            date_node = TreeNode(label="date", is_content=False)
            date_node.add_child(TreeNode(label=date_val, is_content=True))
            event_node.add_child(date_node)

        establishment.add_child(event_node)

    _attach(root, establishment)


def _parse_percentage_list(text: str) -> list[tuple[str, str]]:
    entries = []
    pattern = re.compile(r'(\d+\.?\d*)\s*%\s*([^0-9%]+?)(?=\s*\d+\.?\d*\s*%|$)')
    for match in pattern.finditer(text):
        pct = match.group(1).strip() + "%"
        name = re.sub(r'\s+', ' ', match.group(2)).strip().strip("—–-").strip()
        if name:
            entries.append((name, pct))
    return entries


def _group_percentage_field(root: TreeNode, field_label: str) -> None:
    node = _find_child(root, field_label)
    if node is None:
        return

    raw = " ".join(c.label for c in node.children if c.is_content)
    if not raw:
        return

    entries = _parse_percentage_list(raw)
    if not entries:
        return

    _detach(root, node)

    # Sorted alphabetically so Wikipedia text order does not generate
    # spurious edit distance between countries with identical compositions
    entries.sort(key=lambda pair: pair[0].lower())

    group = TreeNode(label=field_label, is_content=False)
    for name, pct in entries:
        entry_node = TreeNode(label=name, is_content=False)
        entry_node.add_child(TreeNode(label=pct, is_content=True))
        group.add_child(entry_node)

    _attach(root, group)


def _group_language_field(root: TreeNode, field_label: str) -> None:
    node = _find_child(root, field_label)
    if node is None:
        return

    raw = " ".join(c.label for c in node.children if c.is_content)
    if not raw:
        return

    _detach(root, node)

    group = TreeNode(label=field_label, is_content=False)
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        group.add_child(TreeNode(label=part, is_content=True))

    _attach(root, group)


def _count_populated_fields(root: TreeNode) -> int:
    return sum(
        1 for child in root.children
        if not child.is_content and child.label != "name"
    )


def _post_process(root: TreeNode) -> TreeNode:
    _group_population(root)
    _group_gdp(root, "gdp_ppp", "gdp_ppp")
    _group_gdp(root, "gdp_nominal", "gdp_nominal")
    _group_index(root, "gini", "gini")
    _group_index(root, "hdi", "hdi")
    _group_establishment(root)
    _group_leaders(root)
    for field in list(LIST_FIELDS):
        _group_percentage_field(root, field)
    for field in list(LANGUAGE_FIELDS):
        _group_language_field(root, field)
    return root


def _is_viable(root: TreeNode) -> bool:
    return _count_populated_fields(root) >= MINIMUM_VIABLE_FIELD_COUNT


def load_tree(country_name: str) -> TreeNode:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No XML file found for country: {country_name} at {filepath}"
        )

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()
    tree = _build_tree(root_element)
    tree = _post_process(tree)

    if not _is_viable(tree):
        raise ValueError(
            f"'{country_name}' has fewer than {MINIMUM_VIABLE_FIELD_COUNT} "
            f"populated fields and cannot be compared reliably."
        )

    return tree


def load_tree_from_file(filepath: str) -> TreeNode:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()
    tree = _build_tree(root_element)
    tree = _post_process(tree)

    if not _is_viable(tree):
        raise ValueError(
            f"'{filepath}' has fewer than {MINIMUM_VIABLE_FIELD_COUNT} "
            f"populated fields and cannot be compared reliably."
        )

    return tree