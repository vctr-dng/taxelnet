import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(keys_df, mo, pl):
    mo.md(
        f"""
        - key pairs: **{keys_df.height}**
        - key pairs with several real rows: **{keys_df.filter(pl.col("n_real") > 1).height}**
        - key pairs spanning several datasets: **{keys_df.filter(pl.col("n_dataset") > 1).height}**
        - key pairs whose simulation rows differ: **{keys_df.filter(pl.col("n_distinct_sim") > 1).height}**
        """
    )


@app.cell
def _(keys_df, mo):
    # `n_real` = number of real rows (repeats) for that key pair; each has its own simulation row
    key_table = mo.ui.table(
        keys_df.drop("n_dataset").sort("n_real", descending=True),
        selection="single",
        initial_selection=[0],
        label="Select a (force, Path) key pair",
    )
    key_table
    return (key_table,)


@app.cell
def _(mo):
    max_shown = mo.ui.slider(1, 24, value=8, label="max rows shown")
    max_shown
    return (max_shown,)


@app.cell(hide_code=True, expand_output=True)
def _(
    NODE_COLS,
    SIM_GRID_SHAPE,
    TAXEL_GRID_SHAPE,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    key_table,
    max_shown,
    mo,
    plt,
    real_df,
    sim_df,
    taxel_columns,
):
    # each real row (bottom) under its own simulation row (top): sim rows are not assumed identical
    mo.stop(key_table.value.is_empty(), mo.md("Select a key pair in the table above."))
    key = key_table.value.row(0, named=True)
    rows = key["rows"][: max_shown.value]
    nx, ny = SIM_GRID_SHAPE

    sim = sim_df[rows, NODE_COLS].to_numpy().astype(float).reshape(-1, nx, ny)
    real = (
        real_df[rows, taxel_columns()]
        .to_numpy()
        .astype(float)
        .reshape(-1, *TAXEL_GRID_SHAPE)
    )

    n_cols = min(6, len(rows))
    n_pairs = -(-len(rows) // n_cols)
    fig, axes = plt.subplots(
        2 * n_pairs,
        n_cols,
        figsize=(2.6 * n_cols + 1.5, 6.5 * n_pairs + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat:
        ax.axis("off")
    dx, dy = (X_MAX - X_MIN) / (nx - 1), (Y_MAX - Y_MIN) / (ny - 1)
    extent = (Y_MAX + dy / 2, Y_MIN - dy / 2, X_MIN - dx / 2, X_MAX + dx / 2)
    for i, row in enumerate(rows):
        ax_s, ax_r = (
            axes[2 * (i // n_cols), i % n_cols],
            axes[2 * (i // n_cols) + 1, i % n_cols],
        )
        # common scales so repeats are comparable
        im_s = ax_s.imshow(
            sim[i],
            cmap="viridis",
            origin="upper",
            extent=extent,
            vmin=sim.min(),
            vmax=sim.max(),
        )
        im_r = ax_r.imshow(real[i], cmap="viridis", vmin=real.min(), vmax=real.max())
        ax_s.set_title(f"sim row {row}", fontsize=8)
        ax_r.set_title(f"real row {row}", fontsize=8)
    fig.colorbar(
        im_s,
        ax=axes[0::2].ravel().tolist(),
        label="Deformation (mm), + = pushed down",
        shrink=0.6,
    )
    fig.colorbar(im_r, ax=axes[1::2].ravel().tolist(), label="Taxel value", shrink=0.6)

    shown = f"{len(rows)}/{key['n_real']}"
    fig.suptitle(
        f"{key['Path']} - {key['force']:g} N - dataset {key['dataset']} - rows shown: {shown}"
        f" - distinct sim rows: {key['n_distinct_sim']}"
    )
    fig


@app.cell(column=1)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from taxelnet.io_csv import (
        KEY,
        META_COLS,
        REAL_CSV,
        SIM_CSV,
        SIM_GRID_SHAPE,
        TAXEL_GRID_SHAPE,
        X_MAX,
        X_MIN,
        Y_MAX,
        Y_MIN,
        read_matched_csv,
        taxel_columns,
    )

    return (
        KEY,
        META_COLS,
        REAL_CSV,
        SIM_CSV,
        SIM_GRID_SHAPE,
        TAXEL_GRID_SHAPE,
        X_MAX,
        X_MIN,
        Y_MAX,
        Y_MIN,
        mo,
        pl,
        plt,
        read_matched_csv,
        taxel_columns,
    )


@app.cell
def _(META_COLS, REAL_CSV, SIM_CSV, mo, read_matched_csv):
    data_dir = mo.notebook_dir().parent / "data" / "Indenters"
    real_df = read_matched_csv(data_dir / REAL_CSV)
    sim_df = read_matched_csv(data_dir / SIM_CSV)
    NODE_COLS = [c for c in sim_df.columns if c not in META_COLS]  # n195 ... n14899

    # The two files are row-aligned: row i of sim and row i of real are the same press
    assert real_df.select(META_COLS).equals(sim_df.select(META_COLS))
    return NODE_COLS, real_df, sim_df


@app.cell
def _(KEY, NODE_COLS, pl, real_df, sim_df):
    # "dataset" is constant within a key pair, so repeated presses of a key pair have nothing but
    # their row number to tell them apart. Each repeat keeps its own simulation row.
    sim_hash = sim_df.select(pl.concat_list(NODE_COLS).hash().alias("sim_hash"))
    keys_df = (
        real_df.select(*KEY, "dataset")
        .with_columns(sim_hash)
        .with_row_index("row")
        .group_by(KEY, maintain_order=True)
        .agg(
            pl.col("dataset").n_unique().alias("n_dataset"),
            pl.col("dataset").first(),
            pl.len().alias("n_real"),
            pl.col("sim_hash").n_unique().alias("n_distinct_sim"),
            pl.col("row"),
        )
        .rename({"row": "rows"})
        .sort(KEY)
    )
    return (keys_df,)


if __name__ == "__main__":
    app.run()
