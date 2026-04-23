from lst_downscaling.cli import _parse_int_list


def test_parse_int_list_accepts_comma_separated_values():
    assert _parse_int_list("6,7,8", argument_name="--months") == (6, 7, 8)

