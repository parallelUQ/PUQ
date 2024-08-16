import numpy as np
import matplotlib.pyplot as plt


def plotline(df, methods, rep_no, w=2, b=1, s="banana", ylim=[0.000001, 1], idstart=0):
    colors = ["red", "green", "blue", "cyan", "magenta"]
    markers = ["o", "+", "*", "D", "v"]
    linestyles = [(0, ()), (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (5, 10))]
    fig, ax = plt.subplots()

    idmse = 210
    maxval = 0
    minval = 1
    ft = 20
    for m_id, m in enumerate(methods):
        dfnnew = df[df["methods"] == m]
        dfnnnew = dfnnew[dfnnew["theta_id"] <= idmse]

        mslist = []
        sdlist = []
        for i in range(idmse + 1):
            mn = dfnnnew[dfnnnew["theta_id"] == i]["TV"].mean()
            sd = dfnnnew[dfnnnew["theta_id"] == i]["TV"].std()
            mslist.append(mn)
            sdlist.append(1.96 * sd / np.sqrt(rep_no))

        ax.plot(
            range(0, len(mslist)),
            mslist,
            marker=markers[m_id],
            color=colors[m_id],
            ls=linestyles[m_id],
            markersize=10,
            markevery=40,
            linewidth=2,
            label=m,
        )

        maxval = max(maxval, max(mslist))
        minval = min(minval, min(mslist))

    ax.set_ylim((minval, maxval))
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2, prop={"size": 16}
    )  # 16for small
    ax.set_yscale("log")
    ax.set_xlabel("Stage", fontsize=ft)
    ax.set_ylabel("MAD", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.savefig("Figure6_" + s + ".png", bbox_inches="tight")