import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os
data="dataset/global_country_rankings_2000_2026.csv"
OUTPUT="data analysis plots"
os.makedirs(OUTPUT,exist_ok=True)

"""
Metadata & Categorical (4 Columns)
Country: Standardized official country name (217 sovereign nations).
Year: 2000 to 2026.
Region: Macro-region (North America, South America, Europe, Asia, Africa, Oceania).
Economic_Tier: Income classification (1: High Income, 2: Upper Middle, 3: Lower Middle, 4: Low Income).
Global Annual Rankings (11 Rank Columns: Rank 1 = Top Nation)
Happiness_Rank: Global rank in Happiness (1 = Happiest Nation, World Happiness Report).
Global_Hunger_Rank: Global rank in Hunger Reduction (1 = Lowest Hunger, GHI).
Human_Development_Rank: Global rank in Human Development (1 = Highest Development, UN HDI).
GDP_Per_Capita_Rank: Global rank in GDP per Capita Wealth (1 = Highest Wealth, World Bank).
Life_Expectancy_Rank: Global rank in Life Expectancy (1 = Longest Life, World Bank).
Corruption_Perception_Rank: Global rank in Clean Governance (1 = Least Corrupt, CPI).
Democracy_Rank: Global rank in Democracy (1 = Most Democratic, EIU).
Gini_Rank: Global rank in Income Equality (1 = Lowest Inequality, World Bank Gini).
Press_Freedom_Rank: Global rank in Press Freedom (1 = Most Free Press, RSF).
Global_Peace_Rank: Global rank in Peacefulness (1 = Most Peaceful, IEP).
Environmental_Performance_Rank: Global rank in Environmental Health (1 = Healthiest Environment, Yale EPI).
"""

# LOAD DATA
def load_data(data):
    df=pd.read_csv(data)
    return df

# INSPECT DATA
def inspect_data(df):
    print("\n--- HEAD ---")
    print(df.head())
    """
    --- HEAD ---
Country  Year Region  Economic_Tier  Happiness_Rank  ...  Democracy_Rank  Gini_Rank  Press_Freedom_Rank  Global_Peace_Rank  Environmental_Performance_Rank
0  Afghanistan  2000   Asia              4             195  ...             215        183                 214                216                             212
1  Afghanistan  2001   Asia              4             198  ...             215        180                 214                217                             214
2  Afghanistan  2002   Asia              4             196  ...             215        174                 214                217                             213
3  Afghanistan  2003   Asia              4             196  ...             215        173                 214                217                             214
4  Afghanistan  2004   Asia              4             194  ...             216        172                 214                217                             212"""
    print("\n--- TAIL ---")
    print(df.tail())
    """
    --- TAIL ---
Country  Year  Region  Economic_Tier  Happiness_Rank  ...  Democracy_Rank  Gini_Rank  Press_Freedom_Rank  Global_Peace_Rank  Environmental_Performance_Rank
5854  Zimbabwe  2022  Africa              3             167  ...             182        208                 162                163                             186
5855  Zimbabwe  2023  Africa              3             167  ...             183        209                 162                162                             186
5856  Zimbabwe  2024  Africa              3             168  ...             183        210                 162                163                             187
5857  Zimbabwe  2025  Africa              3             170  ...             183        210                 162                162                             186
5858  Zimbabwe  2026  Africa              3             167  ...             183        210                 162                162                             187"""

    print("\n--- DESCRIBE ---")
    print(df.describe())
    """
--- DESCRIBE ---
Year  Economic_Tier  Happiness_Rank  Global_Hunger_Rank  ...    Gini_Rank  Press_Freedom_Rank  Global_Peace_Rank  Environmental_Performance_Rank
count  5859.000000    5859.000000     5859.000000         5859.000000  ...  5859.000000         5859.000000        5859.000000                     5859.000000
mean   2013.000000       2.050691      109.000000           96.936849  ...   108.951528          109.000000         109.000000                      109.000000
std       7.789546       1.035058       62.647186           75.533151  ...    62.672195           62.647186          62.647186                       62.647186
min    2000.000000       1.000000        1.000000            1.000000  ...     1.000000            1.000000           1.000000                        1.000000
25%    2006.000000       1.000000       55.000000            1.000000  ...    55.000000           55.000000          55.000000                       55.000000
50%    2013.000000       2.000000      109.000000          109.000000  ...   109.000000          109.000000         109.000000                      109.000000
75%    2020.000000       3.000000      163.000000          163.000000  ...   163.000000          163.000000         163.000000                      163.000000
max    2026.000000       4.000000      217.000000          217.000000  ...   217.000000          217.000000         217.000000                      217.000000
"""
    print("\n--- INFO ---")
    print(df.info())
    """
    --- INFO ---
    <class 'pandas.DataFrame'>
    RangeIndex: 5859 entries, 0 to 5858
    Data columns (total 15 columns):
    #   Column                          Non-Null Count  Dtype
    ---  ------                          --------------  -----
    0   Country                         5859 non-null   str  
    1   Year                            5859 non-null   int64
    2   Region                          5859 non-null   str  
    3   Economic_Tier                   5859 non-null   int64
    4   Happiness_Rank                  5859 non-null   int64
    5   Global_Hunger_Rank              5859 non-null   int64
    6   Human_Development_Rank          5859 non-null   int64
    7   GDP_Per_Capita_Rank             5859 non-null   int64
    8   Life_Expectancy_Rank            5859 non-null   int64
    9   Corruption_Perception_Rank      5859 non-null   int64
    10  Democracy_Rank                  5859 non-null   int64
    11  Gini_Rank                       5859 non-null   int64
    12  Press_Freedom_Rank              5859 non-null   int64
    13  Global_Peace_Rank               5859 non-null   int64
    14  Environmental_Performance_Rank  5859 non-null   int64"""

    print("\n--- COLUMNS ---")
    print(df.columns)
    """
    --- COLUMNS ---
    Index(['Country', 'Year', 'Region', 'Economic_Tier', 'Happiness_Rank',
        'Global_Hunger_Rank', 'Human_Development_Rank', 'GDP_Per_Capita_Rank',
        'Life_Expectancy_Rank', 'Corruption_Perception_Rank', 'Democracy_Rank',
        'Gini_Rank', 'Press_Freedom_Rank', 'Global_Peace_Rank',
        'Environmental_Performance_Rank'],
        dtype='str')"""

    print("\n--- DUPLICATED ---")
    print(df.duplicated().sum())
    """
    --- DUPLICATED ---
    0
    """

    print("\n--- ISNULL ---")
    print(df.isnull().sum()) # nonull entry

# CORRELATION HEATMAP
def correlation_heatmap_analysis(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(13,10))
    sb.heatmap(corr,annot=True,cmap="coolwarm")
    plt.title("CORRALETION HEATMAP")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/correlation_heatmap_analysis.png")
    plt.show()

# ANALYSIS PLOTS
#-----------------------------------------------------
def Analysis_by_Regions(df):
    region_entries=df["Region"].value_counts()
    sb.barplot(x=region_entries.index,y=region_entries.values)
    plt.title("Number of Entries by Regions")
    plt.xlabel("Regions")
    plt.ylabel("Number of Entries")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_by_Regions.png")
    plt.show()

def Analysis_of_Happiness_rank_by_Regions(df):
    happiness_by_region=df.groupby("Region")["Happiness_Rank"].value_counts().reset_index(name="count")
    plt.figure(figsize=(15,6))
    sb.scatterplot(data=happiness_by_region, x="Happiness_Rank",y="count",hue="Region")
    plt.xlim(0,220)
    plt.title("Happiness Rank by Regions")
    plt.xlabel("Happiness Rank")
    plt.ylabel("Number of Entries")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Happiness_rank_by_Regions.png")
    plt.show()

def Analysis_of_Happiness_rank_by_Economic_Tier(df):
    sb.boxplot(data=df, y="Happiness_Rank",x="Economic_Tier",color="green")
    plt.title("Happiness Rank by Economic Tier")
    plt.xlabel("Economic Tier")
    plt.ylabel("Happiness Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Happiness_rank_by_Economic_Tier.png")
    plt.show()

def Analysis_of_Environmental_Performance_Rank_by_Happiness_rank(df):
    plt.scatter(data=df,x="Environmental_Performance_Rank",y="Happiness_Rank",color="#73B175",s=2)
    func(df["Environmental_Performance_Rank"],df["Happiness_Rank"],1,220,1000,"#2A5D39")
    plt.title("Environmental Performance Rank by Happiness Rank")
    plt.xlabel("Environmental Performance_Rank")
    plt.ylabel("Happiness Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Happiness_rank_by_Environmental_Performance_Rank.png")
    plt.show()

def Analysis_of_Human_Development_Rank_by_Global_Hunger_Rank(df):

    plt.scatter(data=df,x="Human_Development_Rank",y="Global_Hunger_Rank",color="#B873A5",s=2)
    func(df["Human_Development_Rank"],df["Global_Hunger_Rank"],1,210,1000,"#4D0A4F")
    plt.title("Human Development Rank by Global Hunger Rank")
    plt.xlabel("Human Development")
    plt.ylabel("Global Hunger Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Human_Development_Rank_by_Global_Hunger_Rank.png")
    plt.show()

def Analysis_of_Economic_Tier_by_Democracy_Rank(df):
    sb.boxplot(data=df,y="Democracy_Rank",x="Economic_Tier",color="#149393")
    plt.title("Economic Tier by Democracy Rank")
    plt.ylabel("Democracy Rank")
    plt.xlabel("Economic Tier")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Economic_Tier_by_Democracy_Rank.png")
    plt.show()

def Analysis_of_Economic_Tier_by_Press_Freedom_Rank(df):
    sb.boxplot(data=df, y="Press_Freedom_Rank",x="Economic_Tier",color="#62429C")
    plt.title("Economic Tier by Press Freedom Rank")
    plt.xlabel("Economic Tier")
    plt.ylabel("Press Freedom Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Economic_Tier_by_Press_Freedom_Rank.png")
    plt.show()


def Analysis_of_Human_Development_Rank_by_Life_Expectancy_Rank(df):

    plt.scatter(data=df,x="Human_Development_Rank",y="Life_Expectancy_Rank",color="#F0DB1E",s=1)
    func(df["Human_Development_Rank"],df["Life_Expectancy_Rank"],1,210,1000,"#A2461C")
    plt.title("Human Development Rank by Life Expectancy Rank")
    plt.xlabel("Human Development")
    plt.ylabel("Life Expectancy Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Human_Development_Rank_by_Life_Expectancy_Rank.png")
    plt.show()

def Analysis_of_Corruption_Perception_Rank_by_Environmental_Performance_Rank(df):

    plt.scatter(data=df,x="Corruption_Perception_Rank",y="Environmental_Performance_Rank",color="#74DE71",s=1)
    func(df["Corruption_Perception_Rank"],df["Environmental_Performance_Rank"],1,210,1000,"#29522D")
    plt.title("Corruption Perception Rank by Environmental Performance Rank")
    plt.xlabel("Corruption Perception Rank")
    plt.ylabel("Environmental Performance Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Corruption_Perception_Rank_by_Environmental_Performance_Rank.png")
    plt.show()

def Analysis_of_Press_Freedom_Rank_by_Democracy_Rank(df):

    plt.scatter(data=df,x="Press_Freedom_Rank",y="Democracy_Rank",color="#71BADE",s=1)
    func(df["Press_Freedom_Rank"],df["Democracy_Rank"],1,220,1000,"#0C5A8A")
    plt.title("Press Freedom Rank by Democracy Rank")
    plt.xlabel("Press Freedom Rank")
    plt.ylabel("Democracy Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Press_Freedom_Rank_by_Democracy_Rank.png")
    plt.show()

def Analysis_of_Economic_Tier_by_Global_Peace_Rank(df):
    sb.boxplot(data=df,y="Global_Peace_Rank",x="Economic_Tier",color="#D3A313")
    plt.title("Economic Tier by Global Peace Rank")
    plt.xlabel("Economic Tier")
    plt.ylabel("Global Peace Rank")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/Analysis_of_Economic_Tier_by_Global_Peace_Rank.png")
    plt.show()

# FUNC FOR VISUALIZATION
def func(x,y,start,finish,section,colour):
    my_model=np.poly1d(np.polyfit(x,y,3))
    my_line=np.linspace(start,finish,section)
    plt.plot(my_line,my_model(my_line),color=colour)

# MAIN PIPELINE
def main():
    df=load_data(data)
    inspect_data(df)                                           
    # There is no need to clean the dataset.
    
    correlation_heatmap_analysis(df)                              

    Analysis_by_Regions(df)
    # Given that there is an equal number of records for each year, we can identify the continents with the highest and lowest number of country entries in the dataset by examining the data counts for each continent in this table.

    Analysis_of_Happiness_rank_by_Regions(df)
    Analysis_of_Environmental_Performance_Rank_by_Happiness_rank(df)
    Analysis_of_Happiness_rank_by_Economic_Tier(df)
    Analysis_of_Human_Development_Rank_by_Global_Hunger_Rank(df)
    Analysis_of_Economic_Tier_by_Democracy_Rank(df)
    Analysis_of_Economic_Tier_by_Press_Freedom_Rank(df)
    Analysis_of_Human_Development_Rank_by_Life_Expectancy_Rank(df)
    Analysis_of_Press_Freedom_Rank_by_Democracy_Rank(df)
    Analysis_of_Economic_Tier_by_Global_Peace_Rank(df)

# RUN
if __name__=="__main__":
    main()