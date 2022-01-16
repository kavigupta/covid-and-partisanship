def compute_overall(census_over_65, census_under_18, vaxx_18to64, vaxx_65up):
    census_over_18 = 1 - census_under_18
    census_18to64 = census_over_18 - census_over_65
    return (vaxx_18to64 * census_18to64 + vaxx_65up * census_over_65) / census_over_18


def join_and(values):
    values = list(values)
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0] + " and " + values[1]
    raise NotImplementedError()
