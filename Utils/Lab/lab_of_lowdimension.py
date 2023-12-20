import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from umap import UMAP
from Utils.paths import *
import patchworklib as pw
import pandas as pd


# 定义一个函数，用于执行整个流程并绘制图表
def visualize_clustering(num_clusters=None):
    # 生成随机数量的核心点
    if num_clusters is None:
        num_clusters = np.random.randint(2, 20)

    # 生成以随机数量的核心点为中心的团簇数据作为示例
    num_samples = 300
    dimensions = 3

    np.random.seed(42)

    # 生成围绕随机数量的核心点分布的数据
    vectors_list = np.zeros((0, dimensions))
    for _ in range(num_clusters):
        center = 10 * np.random.rand(dimensions)  # 随机化核心点的位置
        cluster = center + np.random.randn(num_samples // num_clusters, dimensions)
        vectors_list = np.vstack((vectors_list, cluster))

    # 使用 PCA 将高维向量降至二维
    pca = PCA(n_components=2, random_state=42)
    vectors_pca = pca.fit_transform(vectors_list)

    # 使用 t-SNE 将高维向量降至二维
    tsne = TSNE(n_components=2, random_state=42)
    vectors_tsne = tsne.fit_transform(vectors_list)

    # 使用 MDS 将高维向量降至二维
    mds = MDS(n_components=2, random_state=42)
    vectors_mds = mds.fit_transform(vectors_list)

    # 使用 Isomap 将高维向量降至二维
    isomap = Isomap(n_components=2)
    vectors_isomap = isomap.fit_transform(vectors_list)

    # 使用 UMAP 将高维向量降至二维
    umap = UMAP(n_components=2, random_state=42)
    vectors_umap = umap.fit_transform(vectors_list)

    # 创建一个 DataFrame 以便使用 Seaborn 进行可视化
    pca_data = {'Dimension 1': vectors_pca[:, 0], 'Dimension 2': vectors_pca[:, 1]}
    tsne_data = {'Dimension 1': vectors_tsne[:, 0], 'Dimension 2': vectors_tsne[:, 1]}
    mds_data = {'Dimension 1': vectors_mds[:, 0], 'Dimension 2': vectors_mds[:, 1]}
    isomap_data = {'Dimension 1': vectors_isomap[:, 0], 'Dimension 2': vectors_isomap[:, 1]}
    umap_data = {'Dimension 1': vectors_umap[:, 0], 'Dimension 2': vectors_umap[:, 1]}
    df_pca = pd.DataFrame(pca_data)
    df_tsne = pd.DataFrame(tsne_data)
    df_mds = pd.DataFrame(mds_data)
    df_isomap = pd.DataFrame(isomap_data)
    df_umap = pd.DataFrame(umap_data)

    # 绘制各种降维方法的图表
    # plt.figure(figsize=(18, 12))

    # 绘制四种降维方法的子图
    ax1 = pw.Brick(figsize=(3, 2))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df_pca, ax=ax1)
    ax1.set_title(f'PCA k={num_clusters}')

    # plt.subplot(243)
    ax2 = pw.Brick(figsize=(3, 2))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df_tsne, ax=ax2)
    ax2.set_title(f't-SNE k={num_clusters}')

    ax3 = pw.Brick(figsize=(3, 2))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df_mds, ax=ax3)
    ax3.set_title(f'MDS k={num_clusters}')

    ax4 = pw.Brick(figsize=(3, 2))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df_isomap, ax=ax4)
    ax4.set_title(f'Isomap k={num_clusters}')

    ax5 = pw.Brick(figsize=(3, 2))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df_umap, ax=ax5)
    ax5.set_title(f'UMAP k={num_clusters}')
    axAll = ax1 | ax2 | ax3 | ax4 | ax5
    # axAll.savefig("k.png")
    return axAll


def main():
    FIG = None
    # 运行流程多次
    for id, i in enumerate(range(2, 20, 2)):  # 这里设置运行次数，比如设置为 2 次
        print(f"Iteration {id + 1}:")
        if FIG is None:
            FIG = visualize_clustering(i)
        else:
            FIG /= visualize_clustering(i)
    FIG.savefig(os.path.join(LabCachePath, "降维算法比较.png"))
    print(f"save picture to {os.path.join(LabCachePath, '降维算法比较.png')}")
