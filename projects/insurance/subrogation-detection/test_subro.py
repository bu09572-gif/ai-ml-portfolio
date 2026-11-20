import data
import pandas as pd
import subro as s


def toy():
    v = pd.DataFrame({
        "unique_id": [1, 2, 3, 4], "collision_id": [10, 10, 20, 20], "vehicle_year": [2015, 2020, 2010, 1800],
        "pre_crash": ["Parked", "Backing", "Going Straight Ahead", "Making Left Turn"],
        "point_of_impact": ["Center Back End", "Center Front End", "Left Front Bumper", "Right Side Doors"],
        "vehicle_damage": ["Center Back End", "Center Front End", "Left Front Bumper", "Right Side Doors"],
        "vehicle_type": ["Sedan"] * 4, "travel_direction": ["North"] * 4, "vehicle_occupants": [0, 1, 1, 2],
        "contributing_factor_1": ["Unspecified", "Backing Unsafely", "Driver Inattention/Distraction", "Failure to Yield Right-of-Way"],
        "contributing_factor_2": ["Unspecified"] * 4})
    c = pd.DataFrame({"collision_id": [10, 20], "crash_date": ["2023-03-01T00:00:00.000", "2023-11-02T00:00:00.000"],
                      "crash_time": ["8:30", "17:05"], "borough": ["QUEENS", None], "number_of_persons_injured": [0, 1]})
    return c, v


def test_two_claims_per_crash_and_labels():
    c = s.claims(*toy())
    assert len(c) == 4
    parked = c[c.our_pre_crash == "Parked"].iloc[0]
    assert parked.recovery == 1 and parked.labelled             # the other car backed into us
    backer = c[c.our_pre_crash == "Backing"].iloc[0]
    assert backer.recovery == 0
    both = c[c.collision_id == 20]
    assert both.recovery.sum() == 0                             # both drivers at fault: shared, no clean recovery
    assert c.our_vehicle_age.isna().sum() == 1                  # 1800 is not a real model year


def test_rule():
    c = pd.DataFrame({"our_pre_crash": ["Parked", "Stopped in Traffic", "Stopped in Traffic", "Going Straight Ahead"],
                      "our_impact": ["Left Side Doors", "Center Back End", "Center Front End", "Center Back End"]})
    assert s.rule(c).tolist() == [1, 1, 0, 0]


def test_features_never_include_fault():
    c = s.claims(*toy())
    assert not any("fault" in col or "factor" in col for col in s.x(c).columns)


def test_real_data_shape():
    assert len(data.vehicles()) == 194968 and len(data.crashes()) == 96608
