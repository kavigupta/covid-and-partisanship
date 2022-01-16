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
    with open("tables/topline.html", "w") as f:
        f.write(render_perc_table(get_topline(adjusted, all_data.pop)))
    with open("tables/electoral_effect.html", "w") as f:
        f.write(
            render_perc_table(
                compute_electoral_effect(
                    adjusted, 1e6, all_data.dem_share, all_data.pop, all_data.data
                )
            )
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
        "Among Vaccinated People",
        "images/vaccinated.svg",
        use_turnout=True,
    )
    plot_election(
        all_data.data,
        adjusted[:, 1],
        "Among Unaccinated People",
        "images/unvaccinated.svg",
        use_turnout=True,
    )
    plot_all_four(
        all_data.data, "Most Present Type of Person", "images/all_four.svg", adjusted
    )


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


if __name__ == "__main__":
    main()
