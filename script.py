import csv
import re
import sys
import tempfile
import json

import electiondata as e
from permacache import permacache
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
import fire

import sklearn.linear_model

sys.path.insert(0, "/home/kavi/2024-bot/mapmaker/")

from mapmaker.data import data_by_year
from mapmaker.colors import Profile, get_color, DEFAULT_CREDIT
from mapmaker.stitch_map import produce_entire_map, produce_entire_map_generic
from mapmaker.mapper import USAPresidencyBaseMap
from mapmaker.aggregation import get_electoral_vote_by_voteshare

usa_profile = Profile(
    symbol=dict(dem="D", gop="R"),
    hue=dict(dem=2 / 3, gop=1),
    bot_name="bot_2024",
    credit=DEFAULT_CREDIT,
    extra_ec=dict(gop=1),
)

uk_profile = Profile(
    symbol=dict(Con="C", Lab="L", LD="D"),
    name=dict(Con="Conservative", Lab="Labour", LD="Liberal Democratic"),
    hue=dict(Con=2 / 3, Lab=1, LD=1 / 6),
    bot_name="bot_2024",
    credit=DEFAULT_CREDIT,
)

uk_profile_2_party = Profile(
    symbol=dict(Con="C", Lab="L"),
    name=dict(Con="Conservative", Lab="Labour and Liberal Democrat Unity"),
    hue=dict(Con=2 / 3, Lab=1),
    bot_name="bot_2024",
    credit=DEFAULT_CREDIT,
)


def load_file(path):
    with open("data/" + path, "r") as f:
        _, _, _, _, header, *rest, _ = list(csv.reader(f))
    result = pd.DataFrame(rest, columns=header)
    f = tempfile.mktemp()
    result.to_csv(f, index=False)
    result = pd.read_csv(f)
    result = result[result.ConstituencyName == result.ConstituencyName]
    return result.set_index("ConstituencyName")


def uk_density():
    with open("data/uk-shapefile.geojson") as f:
        uk_geojson = json.load(f)
    area_uk_sqkm = {
        feat["properties"]["pcon19nm"]: feat["properties"]["st_areashape"] / 1000 ** 2
        for feat in uk_geojson["features"]
    }
    density = load_file("race.csv")[["CONLevelALL"]]
    density["area"] = density.index.map(
        lambda x: area_uk_sqkm[
            x.replace("ô", "o").replace("Pembrokeshire South", "South Pembrokeshire")
        ]
    )
    density["log_density"] = np.log(density.CONLevelALL / density.area)
    return density[["log_density"]]


def load_all_uk():
    britain_race = (
        load_file("race.csv")[["CON%W", "CON%AS", "CON%BB"]].rename(
            columns={"CON%W": "white %", "CON%AS": "asian %", "CON%BB": "black %"}
        )
        / 100
    )
    britain_race["hispanic %"] = 0
    britain_edu = (
        load_file("education.csv")[["CON%Lev_4_qual"]].rename(
            columns={"CON%Lev_4_qual": "bachelor %"}
        )
        / 100
    )
    britain_religion = load_file("religion.csv")[["CON%Rel_nots", "CON%No_rel"]]
    britain_religion = (
        britain_religion.applymap(lambda x: float(x) if x != "..." else 5) / 100
    )
    britain_religion["total_religious"] = 1 - sum(
        britain_religion[k] for k in britain_religion
    )
    britain_religion = britain_religion[["total_religious"]]

    britain = (
        britain_race.join(britain_edu)
        .join(britain_religion)
        .fillna(0.9)
        .join(uk_density())
    )
    return britain


def predict_uk(p_w, adjust_to_normal):
    britain = load_all_uk()
    britain["predictions"] = np.tanh(p_w.predict(np.array(britain)) + adjust_to_normal)
    return britain


def read_uk_map(britain):
    with open("data/2019UKElectionMap.svg") as f:
        map = f.read()
    backmap = {}
    for name in britain.index:
        original_name = name
        name = {
            "Richmond (Yorks)": "Richmond_Yorks",
            "Weston-Super-Mare": "Weston-super-Mare",
            "Carmarthen West and Pembrokeshire South": "Carmarthen_West_and_South_Pembrokeshire",
            "Carmarthen East and Dinefwr": "Carmarthen_East_And_Dinefwr",
            #         "Penrith and The Border": "Penrith_and_the_Border",
            #         "South Holland and The Deepings": "South_Holland_and_the_Deepings",
        }.get(name, name)
        name = (
            name.replace(" ", "_")
            .replace(",", "_")
            .replace("_The", "_the")
            .replace("ô", "o")
        )
        backmap[name] = original_name
        assert f'name="{name}"' in map
    return map, backmap


def draw_britain_2_party(britain, out):
    britain = britain.copy()
    britain["colors"] = usa_profile.place_on_county_colorscale(
        {
            "dem": 0.5 * (1 + britain["predictions"]),
            "gop": 0.5 * (1 - britain["predictions"]),
        }
    )
    draw_britain(britain, usa_profile, out)


def draw_britain(britain, profile, out):
    map, backmap = read_uk_map(britain)
    out_lines = []
    lines = map.split("\n")
    for line in lines:
        if not line.strip().startswith("<path"):
            out_lines.append(line)
            continue
        name = re.search('name="([^"]+)"', line).group(1)
        name = name.split("-")
        if name[-1] == "1":
            name = name[:-1]
        name = "-".join(name)
        assert name in backmap, name
        color = "#%02x%02x%02x" % tuple(
            get_color(profile.county_colorscale, britain.colors[backmap[name]])
        )
        line = re.sub('class="([^"]+)"', 'class="seat"', line)
        line = re.sub('style="', f'style="fill:{color};', line)
        out_lines.append(line)
    with open(out, "w") as f:
        res = "\n".join(out_lines)
        res = res.replace("black", "white")
        res = res.replace("#000000", "#ffffff")
        f.write(res)


def usa_density_data(version=2):

    geojson = USAPresidencyBaseMap().counties
    area_usa_sqkm = {
        feat["id"]: feat["properties"]["CENSUSAREA"] * 0.621371 ** 2
        for feat in geojson["features"]
    }

    data_overall = pd.read_csv(
        "https://raw.githubusercontent.com/kavigupta/census-downloader/master/outputs/counties_census2020.csv"
    )
    normalizer = e.usa_county_to_fips("STUSAB", alaska_handler=e.alaska.FIVE_REGIONS())
    normalizer.rewrite["chugach census area"] = "copper river census area"
    normalizer.apply_to_df(data_overall, "NAME", "FIPS")
    df = data_overall.groupby("FIPS").sum()[["POP100"]]
    df["area"] = [area_usa_sqkm[x] for x in df.index]
    df["log_density"] = np.log(df.POP100 / df.area)
    return df[["log_density", "POP100"]]


@permacache("uk-prediction/usa_data")
def usa_data(version="usa_data_7"):
    df_whole = data_by_year()[2020]
    df = df_whole.set_index("FIPS")
    usa_stats = df[
        [
            "white %",
            "asian %",
            "black %",
            "hispanic %",
            "bachelor %",
            "total_religious",
            "dem_margin",
            "turnout",
        ]
    ]
    usa_stats = usa_stats.join(usa_density_data()).reset_index()
    del usa_stats["FIPS"]
    dem_margin = usa_stats.pop("dem_margin")
    turnout = usa_stats.pop("turnout")
    population = usa_stats.pop("POP100")
    return df_whole, usa_stats, dem_margin, turnout, population


def process_england_wiki_electoral(table):
    table = table.copy()
    table.columns = [re.match(r"([^\[]+).*", x[1]).group(1) for x in table.columns]
    table = table[table.Constituency != "Total for all constituencies"]
    table = table.fillna(0)
    return table


def england_2019_results():
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/Results_of_the_2019_United_Kingdom_general_election"
    )
    tables = [x for x in tables if "Constituency" in str(x)]
    tables = [process_england_wiki_electoral(table) for table in tables]
    england = max(tables, key=lambda x: x.shape[0])
    england = england[["Constituency", "Con", "Lab", "LD", "Total"]].copy()
    england = england.set_index("Constituency")
    england = england.applymap(int)
    total = england.pop("Total")
    for col in england:
        england[col] /= total
    england.index = [
        (
            x.replace("Birmingham ", "Birmingham, ")
            .replace("Brighton ", "Brighton, ")
            .replace("Liverpool ", "Liverpool, ")
            .replace("Plymouth ", "Plymouth, ")
            .replace("Ealing Southall", "Ealing, Southall")
            .replace("Enfield Southgate", "Enfield, Southgate")
            .replace("Lewisham Deptford", "Lewisham, Deptford")
            .replace("Manchester Gorton", "Manchester, Gorton")
            .replace("Manchester Withington", "Manchester, Withington")
            .replace(
                "Sheffield Brightside and Hillsborough",
                "Sheffield, Brightside and Hillsborough",
            )
            .replace("Sheffield Hallam", "Sheffield, Hallam")
            .replace("Sheffield Heeley", "Sheffield, Heeley")
            .replace("Southampton Itchen", "Southampton, Itchen")
            .replace("Southampton Test", "Southampton, Test")
            .replace("Weston-super-Mare", "Weston-Super-Mare")
        )
        for x in england.index
    ]
    england["Other"] = 1 - sum(england[c] for c in england)
    return england.copy()


@permacache("uk-prediction/train_england_model")
def train_england_model(version=4):
    uk = load_all_uk()
    england = england_2019_results()
    x = np.array(uk.loc[england.index])
    y = np.array(england)

    xt, yt = torch.tensor(x).float(), torch.tensor(y).float()
    model = nn.Sequential(
        nn.Linear(x.shape[1], 100),
        nn.Sigmoid(),
        nn.Linear(100, y.shape[1]),
        nn.LogSoftmax(-1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for i in tqdm.trange(10 ** 4):
        opt.zero_grad()
        loss = -((model(xt) * yt)).mean()
        loss.backward()
        if i % 1000 == 0:
            print(i, loss.item())
        opt.step()
    return model


class AddConservative(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        con, rest = x[:, :1], x[:, 1:]
        con = con + self.amount
        return torch.cat([con, rest], axis=1)


def predict_based_on_england(usa_stats, adjust_amount):
    model = train_england_model().eval()
    if adjust_amount is not None:
        *several, final = list(model)
        model = nn.Sequential(*several, AddConservative(adjust_amount), final)

    england = england_2019_results().copy()
    uk = load_all_uk().copy()

    with torch.no_grad():
        yp_england = (
            model(torch.tensor(np.array(uk.loc[england.index])).float()).exp().numpy()
        )
        yp_uk = model(torch.tensor(np.array(uk)).float()).exp().numpy()
        yp_usa = model(torch.tensor(np.array(usa_stats)).float()).exp().numpy()

    pred_con = (yp_england.argmax(1) == 0).sum()
    actual_con = (np.array(england).argmax(1) == 0).sum()
    print("Predicted con seats", pred_con),
    print("Actual con seats", actual_con)
    full_usa = dict(Con=yp_usa[:, 0], Lab=yp_usa[:, 1], LD=yp_usa[:, 2])
    two_usa = dict(Con=yp_usa[:, 0], Lab=yp_usa[:, 1] + yp_usa[:, 2])
    return pred_con, actual_con, yp_uk, full_usa, two_usa


def get_calibrated_prediction(calibrate_full_usa):
    df_whole, usa_stats, *_ = usa_data()
    low, high = 0, 10
    while True:
        mid = (low + high) / 2
        *_, full_usa, two_usa = predict_based_on_england(usa_stats, mid)
        result = get_electoral_vote_by_voteshare(
            df_whole,
            voteshare_by_party=full_usa if calibrate_full_usa else two_usa,
            turnout=df_whole.turnout,
            basemap=USAPresidencyBaseMap(),
            only_nonclose=False,
        )
        print(low, high, result["Lab"])
        if result["Lab"] > 306:
            low = mid
        elif result["Lab"] < 306:
            high = mid
        if result["Lab"] == 306 or (high - low) < 1e-4:
            break
    return full_usa if calibrate_full_usa else two_usa


def predict_uk_with_usa():

    df_whole, usa_stats, dem_margin, turnout, population = usa_data()

    p_w = sklearn.linear_model.Ridge(alpha=1e-1).fit(
        np.array(usa_stats),
        np.arctanh(np.array(dem_margin)),
        sample_weight=np.array(population).astype(np.float64),
    )
    t_w = sklearn.linear_model.Ridge(alpha=1e-1).fit(
        np.array(usa_stats),
        np.arctanh(np.minimum(turnout, 0.99)),
        sample_weight=np.array(population).astype(np.float64),
    )

    print(p_w.coef_)
    _ = produce_entire_map(
        df_whole,
        title="Pred(race, edu, relig, dens)",
        out_path="images/usa_predicted_from_usa.svg",
        dem_margin=np.tanh(p_w.predict(np.array(usa_stats))),
        turnout=np.tanh(t_w.predict(np.array(usa_stats))),
        basemap=USAPresidencyBaseMap(),
        year=2020,
        use_png=False,
    )
    britain_normal = predict_uk(p_w, 0)
    bot, top = 0, 10
    while True:
        k = (bot + top) / 2
        britain_calibrated = predict_uk(p_w, k)
        amount = (britain_calibrated.predictions < 0).sum()
        print(bot, top, amount)
        if amount < 365:
            top = k
        elif amount > 365:
            bot = k
        else:
            break
    draw_britain_2_party(britain_normal, "images/uk_predicted_from_usa_normal.svg")
    draw_britain_2_party(
        britain_calibrated, "images/uk_predicted_from_usa_calibrated.svg"
    )

    republican = (britain_normal.predictions < 0).sum()
    democrat = 650 - republican
    with open("template_predicted_from_usa.svg") as f:
        svg = f.read()
    svg = svg.replace("$D", str(democrat)).replace("$R", str(republican))
    with open("final_predicted_from_usa.svg", "w") as f:
        f.write(svg)


def plot_election(stat):
    p = (stat[:, 0] - stat[:, 1]) / stat.sum(1)
    t = data.turnout * stat.sum(1)
    _ = produce_entire_map(
        data,
        title="Election among vaccinated people",
        out_path="images/vaccinated.svg",
        dem_margin=p,
        turnout=t,
        basemap=USAPresidencyBaseMap(),
        year=2020,
        use_png=False,
    )
