from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, when, col, round

spark = SparkSession.builder.appName("IPL Analysis").getOrCreate()

# ✅ Load CSVs from your system paths
matches = spark.read.option("header", True).csv(
    r"C:\Users\ayana\OneDrive\Desktop\pyspark_starter\matches_clean.csv", inferSchema=True)
deliveries = spark.read.option("header", True).csv(
    r"C:\Users\ayana\OneDrive\Desktop\pyspark_starter\deliveries_clean.csv", inferSchema=True)

# --------------------------------------------
# 1. 📅 Match Statistics
# --------------------------------------------

print("\n📊 1a. Matches Played Per Season")
matches.groupBy("season").agg(count("*").alias("matches_played")).orderBy("season").show()

print("\n🏆 1b. Wins by Each Team Per Season")
matches.groupBy("season", "winner").agg(count("*").alias("wins")) \
    .orderBy("season", "wins", ascending=[True, False]).show()

print("\n🧠 1c. Toss Decision Effectiveness")
matches_with_toss_result = matches.withColumn(
    "toss_win_match_win", when(col("toss_winner") == col("winner"), 1).otherwise(0)
)

matches_with_toss_result.groupBy("toss_decision").agg(
    count("*").alias("total_matches"),
    sum("toss_win_match_win").alias("wins_after_toss")
).withColumn(
    "win_percentage", round((col("wins_after_toss") / col("total_matches")) * 100, 2)
).show()

# --------------------------------------------
# 3. 🏏 Team Performance
# --------------------------------------------

print("\n📈 3a. Total Matches Played and Won by Each Team")
team_matches = matches.selectExpr("team1 as team").union(matches.selectExpr("team2 as team")) \
    .groupBy("team").agg(count("*").alias("matches_played"))

team_wins = matches.groupBy("winner").agg(count("*").alias("matches_won")) \
    .withColumnRenamed("winner", "team")

team_performance = team_matches.join(team_wins, on="team", how="left") \
    .fillna(0, subset=["matches_won"]) \
    .withColumn("win_percentage", round((col("matches_won") / col("matches_played")) * 100, 2)) \
    .orderBy(col("win_percentage").desc())

team_performance.show(truncate=False)

# --------------------------------------------
# 4. 👤 Player Performance
# --------------------------------------------

print("\n🏅 4a. Top 10 Run Scorers")
deliveries.groupBy("batter") \
    .agg(sum("batsman_runs").alias("total_runs")) \
    .orderBy("total_runs", ascending=False).limit(10).show(truncate=False)

print("\n⚡ 4b. Batter Strike Rates (Min 200 Balls Faced)")
balls_faced = deliveries.groupBy("batter").agg(count("ball").alias("balls"))
runs_scored = deliveries.groupBy("batter").agg(sum("batsman_runs").alias("runs"))

strike_rate_df = balls_faced.join(runs_scored, on="batter") \
    .filter(col("balls") >= 200) \
    .withColumn("strike_rate", round((col("runs") / col("balls")) * 100, 2)) \
    .orderBy("strike_rate", ascending=False)

strike_rate_df.show(10, truncate=False)

print("\n🎯 4c. Top Wicket Takers (Excluding Run Outs)")
valid_dismissals = ['caught', 'bowled', 'lbw', 'caught and bowled', 'stumped', 'hit wicket', 'hitwicket']
wickets_df = deliveries.filter(col("dismissal_kind").isin(valid_dismissals)) \
    .groupBy("bowler") \
    .agg(count("dismissal_kind").alias("wickets")) \
    .orderBy("wickets", ascending=False)

wickets_df.show(10, truncate=False)

# --------------------------------------------
# 5. 🏟️ Venue & Toss Impact
# --------------------------------------------

print("\n📌 5a. Best Venues for Batting First (Win % when batting first)")
bat_first_matches = matches.filter(col("result") == "runs")
bat_first_wins = bat_first_matches.groupBy("venue").agg(count("*").alias("wins"))
total_matches = matches.groupBy("venue").agg(count("*").alias("total"))

bat_first_venue_stats = bat_first_wins.join(total_matches, "venue") \
    .withColumn("bat_first_win_pct", round((col("wins") / col("total")) * 100, 2)) \
    .orderBy(col("bat_first_win_pct").desc())

bat_first_venue_stats.show(10, truncate=False)

print("\n📊 5b. Season-wise Chasing vs Batting First Wins")
bat_first = matches.withColumn("bat_first_win", when(col("result") == "runs", 1).otherwise(0))
chasing = matches.withColumn("chasing_win", when(col("result") == "wickets", 1).otherwise(0))

bat_first_summary = bat_first.groupBy("season").agg(sum("bat_first_win").alias("bat_first_wins"))
chasing_summary = chasing.groupBy("season").agg(sum("chasing_win").alias("chasing_wins"))

match_result_comparison = bat_first_summary.join(chasing_summary, on="season", how="outer").orderBy("season")
match_result_comparison.show()

# --------------------------------------------
# 6. 🚨 Extras and Bowling Economy
# --------------------------------------------

print("\n🧾 6a. Extra Runs Conceded by Each Team (Bowling Team)")
deliveries.groupBy("bowling_team").agg(sum("extra_runs").alias("total_extras")) \
    .orderBy("total_extras", ascending=False).show(truncate=False)

print("\n📉 6b. Economy Rate of Bowlers (Min 300 Balls Bowled)")

# Count balls excluding wides (legal deliveries)
balls_bowled = deliveries.filter((col("extras_type").isNull()) | (col("extras_type") != "wides")) \
    .groupBy("bowler") \
    .agg(count("*").alias("balls"))

# Total runs conceded (including all runs)
runs_given = deliveries.groupBy("bowler").agg(sum("total_runs").alias("runs"))

# Join and calculate economy
economy_df = balls_bowled.join(runs_given, on="bowler") \
    .filter(col("balls") >= 300) \
    .withColumn("economy_rate", round((col("runs") / col("balls")) * 6, 2)) \
    .orderBy("economy_rate")

economy_df.show(10, truncate=False)