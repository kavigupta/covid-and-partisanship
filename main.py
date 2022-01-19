import numpy as np
import pandas as pd
import dataframe_image as dfi

from create import (
    load_data,
    plot_predictions,
    render_perc_table,
    get_topline,
    compute_electoral_effect,
    plot_election,
    plot_all_four,
    run_inference,
    vaccine,
)


def table(all_data, adjusted):
    dfi.export(
        render_perc_table(get_topline(adjusted, all_data.pop)),
        "images/topline.png",
        fontsize=100,
    )
    dfi.export(
        render_perc_table(
            compute_electoral_effect(
                adjusted, 1e6, all_data.dem_share, all_data.pop, all_data.data
            )
        ),
        "images/electoral_effect.png",
        fontsize=50,
    )


def plot_maps(all_data, adjusted):
    plot_election(
        all_data.data,
        adjusted[:, :].sum(-1),
        "Overall",
        "images/overall.svg",
        use_turnout=False,
        profile=vaccine,
    )
    plot_election(
        all_data.data,
        adjusted[:, :, 1],
        "Among Republicans",
        "images/republicans.svg",
        use_turnout=False,
        profile=vaccine,
    )
    plot_election(
        all_data.data,
        adjusted[:, :, 0],
        "Among Democrats",
        "images/democrats.svg",
        use_turnout=False,
        profile=vaccine,
    )
    plot_election(
        all_data.data,
        adjusted[:, 0],
        "Among Vaccinated Adults",
        "images/vaccinated.svg",
        use_turnout=True,
    )
    plot_election(
        all_data.data,
        adjusted[:, 1],
        "Among Unaccinated Adults",
        "images/unvaccinated.svg",
        use_turnout=True,
    )
    plot_all_four(
        all_data.data, "Most Present Type of Person", "images/all_four.svg", adjusted
    )


def make_csv(all_data, adjusted):
    pd.DataFrame(
        adjusted.reshape(-1, 4),
        index=np.where(
            all_data.data["state"] == "Alaska",
            all_data.data.FIPS.apply(lambda x: f"AK 5regions: {x[2:]}, Alaska"),
            all_data.data["county"] + ", " + all_data.data["state"],
        ),
        columns=[
            "Vaccinated Democrats",
            "Vaccinated Republicans",
            "Unvaccinated Democrats",
            "Unvaccinated Republicans",
        ],
    ).to_csv("csvs/inferred_breakdown.csv")


def main():
    all_data = load_data()
    unadjusted, adjusted = run_inference(
        all_data.data,
        all_data.demos,
        all_data.vaxx_share,
        all_data.dem_share,
        all_data.pop,
    )
    plot_predictions(all_data.pop, all_data.dem_share, all_data.vaxx_share, unadjusted)
    table(all_data, adjusted)
    plot_maps(all_data, adjusted)
    make_csv(all_data, adjusted)


if __name__ == "__main__":
    main()
