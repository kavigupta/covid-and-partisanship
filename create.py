from types import SimpleNamespace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tqdm.auto as tqdm
import torch
import torch.nn as nn


from permacache import permacache, stable_hash
import electiondata as e

import sys

sys.path.insert(0, "/home/kavi/2024-bot/mapmaker/")

from mapmaker.colors import Profile, DEFAULT_CREDIT
from mapmaker.data import data_by_year
from mapmaker.generate_image import get_model
from mapmaker.mapper import USAPresidencyBaseMap
from mapmaker.stitch_map import produce_entire_map, produce_entire_map_generic

from utils import compute_overall, join_and

DATE = {"line": "01/14/2022", "disp": "2022-01-14"}

# sourced from https://www.nytimes.com/interactive/2020/us/covid-19-vaccine-doses.html
# using the 18-64 and 65+ numbers from nyt


MANUALLY_INFER_STATES = [
    ("Georgia", "GA", compute_overall(0.143, 0.236, 0.59, 0.81)),
    ("Vermont", "VT", compute_overall(0.20, 0.183, 0.81, 0.95)),
]

CREDIT = (
    DEFAULT_CREDIT
    + ". County vaccination data for "
    + join_and(x for _, x, _ in MANUALLY_INFER_STATES)
    + " as well as some rural counties in CA is inferred from demographics and the state topline."
    + " The exact numbers for vaccinated democrats/unvaccinated democrats/etc are inferred,"
    + " but made to match the total numbers of vaccinated and democrats in each county."
    + f" Data is as of {DATE['disp']}."
)

vaccine = Profile(
    symbol=dict(dem="V", gop="U"),
    name=dict(dem="Vaccinated", gop="Unvaccinated"),
    hue=dict(dem=270 / 360, gop=90 / 360),
    bot_name="bot_2024",
    credit=CREDIT,
    credit_scale=0.5,
    suffix=" Adults",
    value="normalize",
)

all_four = Profile(
    symbol=dict(dv="D", du="d", rv="R", ru="r"),
    name=dict(
        dv="Vaxxed Democrats (D)",
        du="Unvaxxed Democrats (d)",
        rv="Vaxxed Republicans (R)",
        ru="Unvaxxed Republicans (r)",
    ),
    hue=dict(dv=8 / 12, du=5 / 12, rv=1 / 12, ru=11 / 12),
    value="normalize",
    bot_name="bot_2024",
    credit=CREDIT,
    credit_scale=0.5,
    suffix="",
    order=["dv", "du", "rv", "ru"],
    min_saturation=1 / 6,
)


@permacache("covid-effect-elections/load_data")
def load_data(version=1):
    model = get_model(calibrated=False)
    demos = model.get_demographics_by_county(year=2020)
    data = data_by_year()[2020]
    vaxx = get_vaxx_data()
    dem_share = np.array(data.dem_margin / 2 + 0.5)
    vaxx_share = np.array(vaxx.loc[data.FIPS]["fv_adults"])
    pop = np.array(data.CVAP)
    return SimpleNamespace(
        data=data, demos=demos, dem_share=dem_share, vaxx_share=vaxx_share, pop=pop
    )


def run_inference(data, demos, vaxx_share, dem_share, pop):
    p = train_predictor(demos, vaxx_share, dem_share, pop, steps=10 ** 5)
    out = p(demos).detach().numpy()
    corrected_vaxx_share = fix_interpolated_vaxx_share_to_state_level(
        data, out.sum(2)[:, 0], vaxx_share, pop
    )
    inferred = np.array(
        [correct(d, v, t) for d, v, t in zip(dem_share, corrected_vaxx_share, out)]
    )
    return out, inferred


def get_topline(adjusted, pop):
    matrixi = (adjusted * pop[:, None, None]).sum(0) / pop.sum()
    topline = pd.DataFrame(
        matrixi,
        columns=["Democratic", "Republican"],
        index=["Vaccinated", "Unvaccinated"],
    )
    topline.loc["Total"] = topline.sum()
    topline["Total"] = topline.Democratic + topline.Republican
    return topline


def get_population():
    population = pd.read_csv(
        "https://raw.githubusercontent.com/kavigupta/census-downloader/master/outputs/counties_census2020.csv"
    )
    normalizer = e.usa_county_to_fips("STUSAB", alaska_handler=e.alaska.FIVE_REGIONS())
    normalizer.rewrite["chugach census area"] = "valdez-cordova census area"
    normalizer.apply_to_df(population, "NAME", "FIPS")
    population = population.groupby("FIPS").sum().P0030001
    population = dict(zip(population.index, population))
    return population


def get_vaxx_data():
    vaxx = raw_vaxx_data()
    for _, state, _ in MANUALLY_INFER_STATES:
        vaxx.loc[
            (vaxx.Recip_State == state),
            "Series_Complete_18PlusPop_Pct",
        ] = 0
    normalizer = e.usa_county_to_fips(
        "Recip_State", alaska_handler=e.alaska.FIVE_REGIONS()
    )
    normalizer.apply_to_df(vaxx, "Recip_County", "FIPS")
    population = get_population()
    vaxx["population"] = vaxx.FIPS.apply(lambda x: population.get(x, np.nan))
    vaxx["fv_adults"] = vaxx.Series_Complete_18PlusPop_Pct / 100 * vaxx.population
    vaxx = vaxx[["FIPS", "fv_adults", "population"]].copy()
    vaxx = vaxx.groupby("FIPS").sum().copy()
    vaxx["fv_adults"] /= vaxx.population
    return vaxx


def raw_vaxx_data():
    with open(
        "/home/kavi/Downloads/COVID-19_Vaccinations_in_the_United_States_County.csv"
    ) as f:
        contents = [next(f)]
        for line in f:
            if not line.startswith(DATE["line"]):
                break
            contents.append(line)

    with open("data/vaxx.csv", "w") as f:
        f.write("\n".join(contents) + "\n")

    vaxx = pd.read_csv("data/vaxx.csv")
    vaxx = vaxx[vaxx.Recip_County != "Unknown County"][
        ["Recip_State", "Recip_County", "Series_Complete_18PlusPop_Pct"]
    ].copy()

    return vaxx


class Predictor(nn.Module):
    def __init__(self, num_demos):
        super().__init__()
        self.param = nn.Parameter(torch.randn(num_demos, 4))

    def forward(self, x):
        x = torch.tensor(x)
        return x.matmul(self.param.softmax(-1)).reshape((-1, 2, 2))

    def predict_each(self, x):
        yall = self(x)
        return yall.sum(1)[:, 0], yall.sum(2)[:, 0]

    def loss(self, x, y1, y2, w):
        y1, y2, w = torch.tensor(y1), torch.tensor(y2), torch.tensor(w)
        yp1, yp2 = self.predict_each(x)
        loss_overall = (yp1 - y1).abs() + (yp2 - y2).abs()
        return (loss_overall * w).sum() / w.sum()


@permacache(
    "covid-effect-elections/train_predictor",
    key_function=dict(
        demos=stable_hash,
        vaxx_share=stable_hash,
        dem_share=stable_hash,
        pop=stable_hash,
    ),
)
def train_predictor(demos, vaxx_share, dem_share, pop, steps):
    mask = vaxx_share > 0

    p = Predictor(demos.shape[1])
    opt = torch.optim.Adam(p.parameters(), lr=4e-4)
    for i in tqdm.trange(steps):
        opt.zero_grad()
        loss = p.loss(demos[mask], dem_share[mask], vaxx_share[mask], pop[mask])
        loss.backward()
        if i % 1000 == 0:
            print(i, loss.item())
        opt.step()

    return p


def correct(d, v, t):
    if v == 0:
        v = t.sum(1)[0]
    A = [
        [1, 0, 1, 0],  # first column
        [0, 1, 0, 1],  # second column
        [1, 1, 0, 0],  # first row
    ]
    b = [
        d,
        1 - d,
        v,
    ]

    return np.mean(
        [
            np.linalg.solve(A + [ev], b + [tv]).reshape(2, 2)
            for tv, ev in zip(t.flatten(), np.eye(4))
        ],
        axis=0,
    )


def plot_election(data, stat, title, path, *, use_turnout, **kwargs):
    p = (stat[:, 0] - stat[:, 1]) / stat.sum(1)
    t = stat.sum(1)
    if use_turnout:
        t = t * data.turnout
    _ = produce_entire_map(
        data,
        title=title,
        out_path=path,
        dem_margin=p,
        turnout=t,
        basemap=USAPresidencyBaseMap(),
        year=2020,
        use_png=True,
        **kwargs,
    )


def plot_all_four(data, title, path, inferred):
    _ = produce_entire_map_generic(
        data,
        title=title,
        out_path=path,
        voteshare_by_party=dict(
            dv=inferred[:, 0, 0],
            du=inferred[:, 1, 0],
            rv=inferred[:, 0, 1],
            ru=inferred[:, 1, 1],
        ),
        turnout=1,
        basemap=USAPresidencyBaseMap(),
        year=2020,
        use_png=True,
        profile=all_four,
    )


def inverse_sigmoid(x):
    return np.arctanh(x * 2 - 1)


def sigmoid(x):
    return np.arctan(x) / 2 + 0.5


def adjust_state_vaxx(state_vaxx, state_pop, vaxxed):
    adjusted = lambda k: sigmoid(inverse_sigmoid(state_vaxx) + k)
    fn = lambda k: (adjusted(k) * state_pop).sum() / state_pop.sum()
    lo, hi = -10, 10
    while (hi - lo) > 1e-10:
        x = (hi + lo) / 2
        y = fn(x)
        if y > vaxxed:
            hi = x
        else:
            lo = x
    return adjusted(x)


def fix_interpolated_vaxx_share_to_state_level(
    data, model_vaxx_share, cdc_vaxx_share, pop
):
    corrected_vaxx_share = np.array(cdc_vaxx_share)
    for state, _, vaxxed in MANUALLY_INFER_STATES:
        [st_idxs] = np.where(data.state == state)
        state_vaxx = adjust_state_vaxx(model_vaxx_share[st_idxs], pop[st_idxs], vaxxed)
        corrected_vaxx_share[st_idxs] = state_vaxx
    return corrected_vaxx_share


def plot_predictions(pop, dem_share, vaxx_share, out):
    def prediction(actual, predicted, statistic, ax):
        actual = actual * 100
        predicted = predicted * 100
        mask = actual > 0
        actual = actual[mask]
        predicted = predicted[mask]

        vs = np.array([actual, predicted])
        limits = np.percentile(vs, 1), vs.max()
        ax.plot(limits, limits, color="black")
        ax.scatter(
            actual, predicted, color="green", marker=".", s=pop[mask] / 3e4, alpha=0.5
        )
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_xlabel(f"Actual {statistic} [%]")
        ax.set_ylabel(f"Predicted {statistic} [%]")
        ax.grid()

    _, axs = plt.subplots(1, 2, dpi=200, figsize=(10, 4))
    prediction(dem_share, out.sum(1)[:, 0], "Dem Voteshare", axs[0])
    prediction(vaxx_share, out.sum(2)[:, 0], "Vaccine Rate", axs[1])
    plt.savefig("images/accuracy.png", facecolor="white")
    plt.close()


def compute_electoral_effect(
    inferred, covid_deaths, dem_share, pop, data, vaccine_protection=15
):

    vaxxed, unvaxxed = (inferred * pop[:, None, None]).sum((0, 2))

    # V * x / P + U * x = D
    # (V / P + U) * x = D
    unvaxxed_rate = covid_deaths / (vaxxed / vaccine_protection + unvaxxed)
    vaxxed_rate = unvaxxed_rate / vaccine_protection
    deaths_by_party = (
        inferred[:, 0] * vaxxed_rate + inferred[:, 1] * unvaxxed_rate
    ) * pop[:, None]

    deaths = pd.DataFrame(
        {
            "D_d": deaths_by_party[:, 0],
            "R_d": deaths_by_party[:, 1],
            "state": np.array(data.state),
            "D": dem_share * pop,
            "R": (1 - dem_share) * pop,
        },
        index=data.FIPS,
    )
    deaths = deaths.groupby("state").sum()
    margin = lambda: (deaths["D"] - deaths["R"]) / (deaths["D"] + deaths["R"])
    deaths["original_margin"] = margin()
    deaths.D -= deaths.D_d
    deaths.R -= deaths.R_d
    deaths["new_margin"] = margin()
    deaths["Shift in Margin"] = deaths.new_margin - deaths.original_margin
    deaths = deaths[np.abs(deaths.new_margin) < 0.10]
    deaths = deaths[["Shift in Margin"]].sort_values("Shift in Margin")[::-1]
    return pd.DataFrame(
        np.array(deaths),
        columns=["Shift in Margin"],
        index=np.array(deaths.index),
    )


def render_perc_table(table):
    return table.style.format("{:.2%}") .set_properties(**{"text-align": "right"})
