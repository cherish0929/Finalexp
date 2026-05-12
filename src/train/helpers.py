"""Small helper predicates for loss functions."""


def _is_vof_field(field_name: str) -> bool:
    """Return True for Volume-of-Fluid fields (alpha.* or gamma.*)."""
    return field_name.startswith("alpha") or field_name.startswith("gamma")
