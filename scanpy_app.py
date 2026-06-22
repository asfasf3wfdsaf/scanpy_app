import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="单细胞分析工具", layout="wide")
st.title("🧬 单细胞分析 - Scanpy 练手工具")

import tempfile
import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(APP_DIR, "pbmc68k_result.h5ad")
TEMP_PATH = os.path.join(tempfile.gettempdir(), "pbmc68k_workflow.h5ad")

st.markdown("""
<style>
.js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": False,
    "dragmode": "pan",
}

PLOTLY_CONFIG_ZOOM = {
    "displayModeBar": False,
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": True,
    "dragmode": "pan",
}


def safe_write_h5ad(adata, path):
    """云端不保存临时文件，直接跳过"""
    pass


def make_umap_fig(adata, color_col, title, height):
    df = pd.DataFrame({
        "UMAP1": adata.obsm["X_umap"][:, 0],
        "UMAP2": adata.obsm["X_umap"][:, 1],
        "Cluster": adata.obs[color_col].astype(str).values,
    })
    n = df["Cluster"].nunique()
    fig = px.scatter(
        df, x="UMAP1", y="UMAP2", color="Cluster",
        title=title, color_discrete_sequence=px.colors.qualitative.Bold[:n],
        hover_data={"UMAP1": True, "UMAP2": True, "Cluster": True},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.85))
    fig.update_layout(
        width=None, height=height,
        font_family="sans-serif", title_font_size=16,
        legend_title_text="Cluster",
        xaxis_title="UMAP1", yaxis_title="UMAP2",
        plot_bgcolor="white", paper_bgcolor="white",
        hoverlabel=dict(font_size=12, bgcolor="white"),
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e0e0e0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e0e0e0", zeroline=False)
    return fig


def make_gene_fig(adata, g, height):
    X = adata[:, g].X
    expr = X.toarray().flatten() if hasattr(X, 'toarray') else np.asarray(X).flatten()
    umap1, umap2 = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
    fig = go.Figure(go.Scatter(
        x=umap1, y=umap2, mode="markers",
        marker=dict(color=expr, colorscale="RdYlBu_r",
                    size=5, opacity=0.85, showscale=True,
                    colorbar=dict(title="表达量", tickformat=".2f")),
        text=[f"基因: {g}<br>表达量: {v:.4f}<br>UMAP1: {x:.3f}<br>UMAP2: {y:.3f}"
              for v, x, y in zip(expr, umap1, umap2)],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        width=None, height=height,
        title=dict(text=f"<b>{g}</b>", font_size=16),
        xaxis_title="UMAP1", yaxis_title="UMAP2",
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="sans-serif",
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e0e0e0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e0e0e0", zeroline=False)
    return fig


def make_rank_genes_fig(adata, top_n, height):
    """差异基因条形图（matplotlib 版本，标签对齐准确）"""
    result = adata.uns["rank_genes_groups"]
    clusters = list(result["names"].dtype.names)
    n_clusters = len(clusters)

    fig, ax = plt.subplots(figsize=(10, max(4, n_clusters * 0.8)))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    bar_height = 0.8 / n_clusters
    y_positions = np.arange(top_n)

    for ci, cluster in enumerate(clusters):
        genes = [result["names"][cluster][i] for i in range(top_n)]
        scores = [result["scores"][cluster][i] for i in range(top_n)]
        offsets = y_positions + (ci - n_clusters / 2 + 0.5) * bar_height
        bars = ax.barh(offsets, scores, height=bar_height * 0.9,
                       color=colors[ci], label=cluster, alpha=0.85)
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{score:.2f}', va='center', fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([result["names"][clusters[0]][i] for i in range(top_n)], fontsize=9)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("各 Cluster 差异基因 Top", fontsize=13, pad=10)
    ax.legend(title="Cluster", fontsize=8, title_fontsize=9)
    ax.set_xlim(0, max([max(result["scores"][c][:top_n]) for c in clusters]) * 1.3)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ============================================================
# Session
# ============================================================
for key in ["adata", "workflow_done", "rank_genes_done", "tab_idx",
            "umap_big", "rg_big"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ("workflow_done", "rank_genes_done", "umap_big", "rg_big") else None
st.session_state.tab_idx = 0

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    st.caption("邻居数：每个细胞找几个最相似的邻居")
    n_neighbors = st.slider("邻居数", 5, 30, 15, 1)
    st.caption("PCA 维数：用于构建邻居图的 PCA 维度数")
    n_pcs = st.slider("PCA 维数", 5, 50, 30, 5)
    st.caption("分辨率：聚类精细度，值越大分群越多")
    resolution = st.slider("Leiden 分辨率", 0.1, 2.0, 0.5, 0.1)
    st.markdown("---")
    st.markdown("**PBMC 常见 Marker 基因**")
    st.markdown("""
    | 细胞类型 | Marker 基因 |
    |---|---|
    | T 细胞 | `CD3D`、`CD3E` |
    | CD8+ T 细胞 | `CD8A` |
    | B 细胞 | `MS4A1`（CD20）、`CD19` |
    | NK 细胞 | `GNLY`、`NKG7` |
    | 树突状细胞 | `FCER1A` |
    """)
    st.markdown("""
    ---
    **图片操作说明**
    - 🖱️ **滚轮**：缩放（仅放大视图）
    - 🤚 **拖动**：平移
    - 🏠 **双击**：复位
    - 📋 **悬停**：显示坐标和数值
    """)

# ============================================================
# Tab 导航
# ============================================================
st.session_state.tab_idx = st.radio(
    "导航",
    [0, 1, 2],
    format_func=lambda x: ["📊 数据加载", "🔬 分析流程", "📈 可视化"][x],
    index=st.session_state.tab_idx,
    label_visibility="collapsed",
    horizontal=True,
)

# ============================================================
# TAB 0: 数据加载
# ============================================================
if st.session_state.tab_idx == 0:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("数据来源")
        data_source = st.radio("选择", ["PBMC 68K（内置，推荐新手）", "从 .h5ad 文件加载"])
    with c2:
        if data_source == "PBMC 68K（内置，推荐新手）":
            st.info("**PBMC 68K**：健康人外周血单核细胞，700细胞×765基因。含 T/B/NK/单核细胞等。")
        else:
            st.info("**加载 .h5ad**：把你的数据文件路径填进来。")
    st.divider()
    st.warning("⚠️ 注意：刷新页面会重置分析进度（数据不会丢，只是需要重新跑分析流程）")
    if st.button("加载数据", type="primary", use_container_width=True):
        if data_source == "PBMC 68K（内置，推荐新手）":
            with st.spinner("加载中..."):
                st.session_state.adata = sc.datasets.pbmc68k_reduced()
                st.session_state.workflow_done = False
                st.session_state.rank_genes_done = False
                safe_write_h5ad(st.session_state.adata, TEMP_PATH)
                st.success(f"✅ 加载完成！{st.session_state.adata.n_obs} 细胞 × {st.session_state.adata.n_vars} 基因")
        else:
            uploaded = st.file_uploader("上传 .h5ad 文件", type=["h5ad"])
            if uploaded is not None:
                with st.spinner("加载中..."):
                    import io
                    adata = sc.read_h5ad(io.BytesIO(uploaded.getvalue()))
                st.session_state.adata = adata
                st.session_state.workflow_done = False
                st.session_state.rank_genes_done = False
                st.success(f"✅ 加载完成！{adata.n_obs} 细胞 × {adata.n_vars} 基因")
    if st.session_state.adata is not None:
        adata = st.session_state.adata
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("细胞数", f"{adata.n_obs:,}")
        m2.metric("基因数（高变异）", f"{adata.n_vars:,}")
        m3.metric("预计算聚类", "有" if "louvain" in adata.obs else "无")
        m4.metric("预计算 UMAP", "有" if "X_umap" in adata.obsm else "无")
        with st.expander("📋 obs 字段说明"):
            st.write(adata.obs.head(3))
            st.markdown("""
            | 字段 | 含义 |
            |---|---|
            | `bulk_labels` | 预设细胞类型（仅供参考） |
            | `n_genes` / `n_counts` | 每个细胞的基因数和分子数 |
            | `percent_mito` | 线粒体基因占比，>20% 可能是死细胞 |
            | `phase` | 细胞周期（S/G2M） |
            """)

# ============================================================
# TAB 1: 分析流程
# ============================================================
elif st.session_state.tab_idx == 1:
    if st.session_state.adata is None:
        st.warning("⚠️ 请先在「数据加载」加载数据")
    else:
        st.markdown("### 流程说明（共 3 步）")
        st.markdown("""
        | 步骤 | 做什么 | 意义 |
        |---|---|---|
        | ① 邻居计算 | 每个细胞找 K 个最相似的邻居 | 相似细胞应聚在一起 |
        | ② UMAP 降维 | 765 维压缩到 2 维可视化 | 人眼可看细胞分布 |
        | ③ Leiden 聚类 | 基于邻居关系自动分群 | 识别细胞类型 |
        """)
        st.info("👈 参数在左侧边栏可调，改完重新点「执行完整流程」")
        st.divider()
        if st.button("▶️ 执行完整流程", type="primary", use_container_width=True):
            adata = st.session_state.adata
            with st.spinner("① 计算邻居关系..."):
                sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
            with st.spinner("② 计算 UMAP..."):
                sc.tl.umap(adata, min_dist=0.3)
            with st.spinner("③ Leiden 聚类..."):
                sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
            st.session_state.adata = adata
            st.session_state.workflow_done = True
            st.session_state.rank_genes_done = False
            safe_write_h5ad(adata, TEMP_PATH)
            st.success("✅ 完成！切换到「可视化」查看结果")
        if st.session_state.workflow_done:
            st.markdown("---")
            st.markdown("### 聚类结果统计")
            st.caption("每行 = 一个群；右列 = 该群细胞数。数量越多，这类细胞在样本中越多。")
            counts = st.session_state.adata.obs["leiden"].value_counts().sort_index()
            st.dataframe(counts.rename("细胞数"))

# ============================================================
# TAB 2: 可视化
# ============================================================
elif st.session_state.tab_idx == 2:
    if st.session_state.adata is None or not st.session_state.workflow_done:
        st.warning("⚠️ 请先加载数据并执行「分析流程」")
    else:
        adata = st.session_state.adata
        plot_type = st.selectbox(
            "图表类型",
            ["📍 UMAP 聚类", "🧬 Marker 基因表达", "🔥 各 Cluster 差异基因"],
            label_visibility="collapsed",
        )

        OPS = "**🖱️ 操作说明**　滚轮缩放（仅放大视图）　|　🤚 拖动平移　|　🏠 双击复位　|　📋 悬停显示数值"

        # ---- UMAP 聚类 ----
        if plot_type == "📍 UMAP 聚类":
            st.markdown("### UMAP 聚类结果")
            st.markdown(OPS)
            fig = make_umap_fig(adata, "leiden", "Leiden Clustering", 550)
            st.plotly_chart(fig, config=PLOTLY_CONFIG)
            st.caption("↑ 悬停查看每个点的 UMAP1、UMAP2 坐标和所在 Cluster")

            left, right = st.columns([1, 4])
            with left:
                if st.button("🔍 全尺寸放大查看", key="btn_umap"):
                    st.session_state.umap_big = True
                if st.session_state.umap_big:
                    if st.button("✕ 关闭全尺寸视图", key="btn_close_umap"):
                        st.session_state.umap_big = False

            if st.session_state.umap_big:
                st.markdown("---")
                st.markdown("### 🔍 UMAP 全尺寸视图")
                st.markdown(OPS)
                fig_big = make_umap_fig(adata, "leiden", "Leiden Clustering — 全尺寸", 750)
                st.plotly_chart(fig_big, config=PLOTLY_CONFIG_ZOOM)
                st.caption("💡 请先点击图片激活，再使用滚轮缩放 | 悬停查看坐标和 Cluster")

            st.markdown("""
            **怎么看？** 点 = 细胞，颜色 = Leiden 分群。距离近的细胞转录组越相似，同一颜色聚集说明可能是同类细胞。
            """)

        # ---- Marker 基因表达 ----
        elif plot_type == "🧬 Marker 基因表达":
            st.markdown("### Marker 基因表达量")
            st.markdown(OPS)
            all_genes = sorted(list(adata.var_names))
            defaults = [g for g in ["CD3D", "CD8A", "MS4A1", "GNLY", "NKG7", "FCER1A"] if g in all_genes]

            # 搜索框
            search = st.text_input("🔍 搜索基因名称", placeholder="输入基因名，如 CD3D")
            if search:
                filtered = [g for g in all_genes if search.lower() in g.lower()]
            else:
                filtered = all_genes

            # 勾选框选择
            st.caption(f"共 {len(all_genes)} 个基因，已过滤 {len(filtered)} 个")
            selected = []
            cols = st.columns(3)
            for i, g in enumerate(filtered[:90]):  # 最多显示90个避免太慢
                with cols[i % 3]:
                    if st.checkbox(g, value=(g in defaults), key=f"gene_{g}"):
                        selected.append(g)

            st.markdown(f"已选 **{len(selected)}** 个基因")
            if not selected:
                st.warning("请至少选择一个基因")
            else:
                expr_cache = {}
                for g in selected:
                    X = adata[:, g].X
                    expr_cache[g] = X.toarray().flatten() if hasattr(X, 'toarray') else np.asarray(X).flatten()
                umap1 = adata.obsm["X_umap"][:, 0]
                umap2 = adata.obsm["X_umap"][:, 1]

                for g in selected:
                    sk = f"gb_{g}"
                    if sk not in st.session_state:
                        st.session_state[sk] = False

                    st.markdown("---")
                    st.markdown(f"#### {g}")

                    left, right = st.columns([4, 1])
                    with left:
                        fig_g = make_gene_fig(adata, g, 500)
                        st.plotly_chart(fig_g, config=PLOTLY_CONFIG)
                        st.caption(f"↑ 悬停查看每个点的表达量 | 图例颜色条表示表达量高低")
                    with right:
                        st.markdown("**🖱️ 操作说明**")
                        st.markdown("滚轮缩放（放大视图） | 拖动平移 | 双击复位 | 悬停显示数值")
                        if st.button("🔍 放大", key=f"btn_expand_{g}"):
                            st.session_state[sk] = True
                        if st.session_state[sk]:
                            if st.button("✕ 关闭", key=f"btn_close_{g}"):
                                st.session_state[sk] = False

                    if st.session_state[sk]:
                        st.markdown(f"##### 🔍 {g} — 全尺寸视图")
                        st.markdown(OPS)
                        fig_big = make_gene_fig(adata, g, 750)
                        st.plotly_chart(fig_big, config=PLOTLY_CONFIG_ZOOM)
                        st.caption("💡 请先点击图片激活，再使用滚轮缩放")

            st.markdown("""
            **基因速查表**
            | 基因 | 高表达 → 可能是 |
            |---|---|
            | `CD3D` / `CD3E` | T 细胞 |
            | `CD8A` | CD8+ T 细胞（杀手 T） |
            | `MS4A1`（CD20）| B 细胞 |
            | `GNLY` / `NKG7` | NK 细胞（自然杀伤）|
            | `FCER1A` | 树突状细胞 |
            """)

        # ---- 差异基因 ----
        elif plot_type == "🔥 各 Cluster 差异基因":
            st.markdown("### 各 Cluster 差异基因")
            st.caption("每个 Cluster 显示表达量最高的 Top 8 基因，基因名在 y 轴，Score 在 x 轴")

            if st.button("🧮 计算差异基因", use_container_width=True):
                with st.spinner("t-test 计算中（请稍候）..."):
                    sc.tl.rank_genes_groups(adata, groupby="leiden", method="t-test")
                    st.session_state.rank_genes_done = True
                    safe_write_h5ad(adata, TEMP_PATH)
                st.success("✅ 计算完成！")

            if st.session_state.rank_genes_done:
                fig = make_rank_genes_fig(adata, top_n=8, height=6)
                st.pyplot(fig)
                st.caption("各 Cluster 差异基因 Top 8，y轴为基因名，x轴为 t-test Score")

                left, right = st.columns([1, 4])
                with left:
                    if st.button("🔍 全尺寸放大查看", key="btn_rg"):
                        st.session_state.rg_big = True
                    if st.session_state.rg_big:
                        if st.button("✕ 关闭全尺寸视图", key="btn_close_rg"):
                            st.session_state.rg_big = False

                if st.session_state.rg_big:
                    st.markdown("---")
                    st.markdown("### 🔍 差异基因 — 全尺寸视图")
                    fig_big = make_rank_genes_fig(adata, top_n=8, height=9)
                    st.pyplot(fig_big)
                    st.caption("各 Cluster 差异基因 Top 8，y轴为基因名，x轴为 t-test Score")

                st.markdown("---")
                st.markdown("### 差异基因列表")
                result = adata.uns["rank_genes_groups"]
                df = pd.DataFrame({
                    f"Cluster {g}": list(result["names"][g][:8])
                    for g in result["names"].dtype.names
                })
                st.dataframe(df)
                st.markdown("""
                **怎么看？** 每个 Cluster 显示表达量最高的基因。如果某已知 marker 在某 Cluster 排第一 → 该群可能是对应细胞类型。
                """)

        st.divider()
        st.markdown("### 💾 保存结果")
        st.caption("点击下载将分析结果保存为 .h5ad 文件")
        import io
        buf = io.BytesIO()
        adata.write_h5ad(buf)
        buf.seek(0)
        st.download_button(
            "📥 下载 .h5ad 结果文件",
            data=buf.getvalue(),
            file_name="scanpy_result.h5ad",
            mime="application/x-h5ad",
            help="下载当前分析结果，包含数据、聚类、UMAP坐标等",
        )
