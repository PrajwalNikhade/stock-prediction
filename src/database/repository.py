"""Database repository helpers for CrowdWisdomTrading."""

import json
import sqlite3
from pathlib import Path
import pandas as pd
from src.database.db import get_connection
from src.utils.time_utils import format_utc, utc_now


class Repository:
    """Unified data-access layer over SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    # --- Price bars ---
    def save_price_bars(self, df: pd.DataFrame) -> int:
        """Upsert price bars. Returns count inserted."""
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO price_bars
                           (asset, timestamp_utc, timeframe, open, high, low, close, volume, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row.get("asset"),
                            row.get("timestamp_utc"),
                            row.get("timeframe"),
                            row.get("open"),
                            row.get("high"),
                            row.get("low"),
                            row.get("close"),
                            row.get("volume"),
                            row.get("source"),
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return count

    def get_price_bars(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Retrieve price bars sorted by timestamp."""
        conn = self._conn()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM price_bars WHERE asset=? AND timeframe=? ORDER BY timestamp_utc",
                conn,
                params=(asset, timeframe),
            )
            return df
        finally:
            conn.close()

    # --- X Posts ---
    def save_raw_posts(self, df: pd.DataFrame) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO raw_x_posts
                           (post_id, asset, timestamp_utc, post_text, author_id,
                            author_username, author_followers, likes, reposts,
                            replies, views, post_url, search_query,
                            collection_timestamp_utc, source, raw_json)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row.get("post_id"),
                            row.get("asset"),
                            row.get("timestamp_utc"),
                            row.get("post_text"),
                            row.get("author_id"),
                            row.get("author_username"),
                            row.get("author_followers"),
                            row.get("likes"),
                            row.get("reposts"),
                            row.get("replies"),
                            row.get("views"),
                            row.get("post_url"),
                            row.get("search_query"),
                            row.get("collection_timestamp_utc"),
                            row.get("source", "apify"),
                            row.get("raw_json"),
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return count

    def get_raw_posts(self, asset: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM raw_x_posts WHERE asset=? ORDER BY timestamp_utc",
                conn,
                params=(asset,),
            )
        finally:
            conn.close()

    def save_processed_posts(self, df: pd.DataFrame) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO processed_x_posts
                           (post_id, asset, timestamp_utc, cleaned_text,
                            vader_compound, vader_pos, vader_neg, vader_neu,
                            transformer_positive, transformer_negative, transformer_neutral,
                            final_sentiment_score, sentiment_label, processing_method,
                            engagement_weight, processing_timestamp, is_relevant)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row.get("post_id"),
                            row.get("asset"),
                            row.get("timestamp_utc"),
                            row.get("cleaned_text"),
                            row.get("vader_compound"),
                            row.get("vader_pos"),
                            row.get("vader_neg"),
                            row.get("vader_neu"),
                            row.get("transformer_positive"),
                            row.get("transformer_negative"),
                            row.get("transformer_neutral"),
                            row.get("final_sentiment_score"),
                            row.get("sentiment_label"),
                            row.get("processing_method"),
                            row.get("engagement_weight"),
                            row.get("processing_timestamp"),
                            row.get("is_relevant", 1),
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return count

    def get_processed_posts(self, asset: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM processed_x_posts WHERE asset=? AND is_relevant=1 ORDER BY timestamp_utc",
                conn,
                params=(asset,),
            )
        finally:
            conn.close()

    # --- News ---
    def save_news(self, df: pd.DataFrame) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO news_articles
                           (news_id, asset, title, snippet, source_name, url,
                            published_at, retrieved_at, provider, sentiment_score, sentiment_label)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row.get("news_id"),
                            row.get("asset"),
                            row.get("title"),
                            row.get("snippet"),
                            row.get("source_name"),
                            row.get("url"),
                            row.get("published_at"),
                            row.get("retrieved_at"),
                            row.get("provider"),
                            row.get("sentiment_score"),
                            row.get("sentiment_label"),
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return count

    def get_news(self, asset: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM news_articles WHERE asset=? ORDER BY published_at DESC",
                conn,
                params=(asset,),
            )
        finally:
            conn.close()

    # --- Sentiment aggregates ---
    def save_sentiment_aggregates(self, df: pd.DataFrame) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO sentiment_aggregates
                           (asset, period_start, timeframe, mean_sentiment, median_sentiment,
                            positive_ratio, negative_ratio, neutral_ratio, net_sentiment,
                            sentiment_std, post_count, relevant_post_count, unique_author_count,
                            likes_sum, reposts_sum, views_sum, engagement_weighted_sentiment)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row.get("asset"),
                            row.get("period_start"),
                            row.get("timeframe"),
                            row.get("mean_sentiment"),
                            row.get("median_sentiment"),
                            row.get("positive_ratio"),
                            row.get("negative_ratio"),
                            row.get("neutral_ratio"),
                            row.get("net_sentiment"),
                            row.get("sentiment_std"),
                            row.get("post_count"),
                            row.get("relevant_post_count"),
                            row.get("unique_author_count"),
                            row.get("likes_sum"),
                            row.get("reposts_sum"),
                            row.get("views_sum"),
                            row.get("engagement_weighted_sentiment"),
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return count

    def get_sentiment_aggregates(self, asset: str, timeframe: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM sentiment_aggregates WHERE asset=? AND timeframe=? ORDER BY period_start",
                conn,
                params=(asset, timeframe),
            )
        finally:
            conn.close()

    # --- Engineered Features ---
    def save_engineered_features(self, df: pd.DataFrame, asset: str, timeframe: str, run_id: str = None) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                ts = str(row.get("timestamp_utc", row.get("feature_timestamp", "")))
                row_dict = row.to_dict()
                # Clean up non-serializable objects
                clean_dict = {}
                for k, v in row_dict.items():
                    if pd.isna(v):
                        clean_dict[k] = None
                    elif hasattr(v, "isoformat"):
                        clean_dict[k] = v.isoformat()
                    elif isinstance(v, (int, float, str, bool)):
                        clean_dict[k] = v
                    else:
                        clean_dict[k] = str(v)
                conn.execute(
                    """INSERT OR REPLACE INTO engineered_features
                       (asset, timestamp_utc, timeframe, feature_json, run_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (asset, ts, timeframe, json.dumps(clean_dict), run_id),
                )
                count += 1
            conn.commit()
        finally:
            conn.close()
        return count

    def get_engineered_features(self, asset: str, timeframe: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM engineered_features WHERE asset=? AND timeframe=? ORDER BY timestamp_utc",
                conn,
                params=(asset, timeframe),
            )
        finally:
            conn.close()

    # --- Forecasts ---
    def save_forecasts(self, records: list[dict], run_id: str = None) -> int:
        conn = self._conn()
        count = 0
        try:
            for r in records:
                conn.execute(
                    """INSERT INTO forecasts
                       (run_id, asset, timeframe, forecast_origin, horizon, target_timestamp,
                        reference_price, predicted_return, predicted_price, model_name)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id or r.get("run_id"),
                        r["asset"], r["timeframe"], r["forecast_origin"],
                        r["horizon"], r.get("target_timestamp"),
                        r.get("reference_price"), r.get("predicted_return"),
                        r.get("predicted_price"), r.get("model_name"),
                    ),
                )
                count += 1
            conn.commit()
        finally:
            conn.close()
        return count

    def get_latest_forecasts(self, asset: str, timeframe: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                """SELECT * FROM forecasts WHERE asset=? AND timeframe=?
                   ORDER BY created_at DESC, horizon ASC LIMIT 6""",
                conn,
                params=(asset, timeframe),
            )
        finally:
            conn.close()

    # --- Ingestion runs ---
    def log_ingestion_run(self, record: dict) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO ingestion_runs
                   (asset, provider, run_type, requested_start, requested_end,
                    actual_start, actual_end, record_count, status, error_message)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["asset"], record["provider"], record["run_type"],
                    record.get("requested_start"), record.get("requested_end"),
                    record.get("actual_start"), record.get("actual_end"),
                    record.get("record_count", 0), record["status"],
                    record.get("error_message"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_ingestion_runs(self, asset: str = None) -> pd.DataFrame:
        conn = self._conn()
        try:
            if asset:
                return pd.read_sql_query(
                    "SELECT * FROM ingestion_runs WHERE asset=? ORDER BY created_at DESC",
                    conn, params=(asset,),
                )
            return pd.read_sql_query(
                "SELECT * FROM ingestion_runs ORDER BY created_at DESC", conn
            )
        finally:
            conn.close()

    # --- Validation results ---
    def save_validation_results(self, records: list[dict], run_id: str = None) -> int:
        conn = self._conn()
        count = 0
        try:
            for r in records:
                conn.execute(
                    """INSERT INTO validation_results
                       (run_id, asset, timeframe, model_name, horizon, fold,
                        metric_name, metric_value, dataset_type)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id or r.get("run_id"),
                        r["asset"], r["timeframe"], r["model_name"],
                        r["horizon"], r.get("fold"),
                        r["metric_name"], r["metric_value"], r["dataset_type"],
                    ),
                )
                count += 1
            conn.commit()
        finally:
            conn.close()
        return count

    def get_validation_results(self, asset: str, timeframe: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM validation_results WHERE asset=? AND timeframe=? ORDER BY created_at DESC",
                conn, params=(asset, timeframe),
            )
        finally:
            conn.close()

    # --- Backtest results ---
    def save_backtest_results(self, df: pd.DataFrame, run_id: str = None) -> int:
        conn = self._conn()
        count = 0
        try:
            for _, row in df.iterrows():
                ts = row.get("timestamp_utc")
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                elif ts is not None:
                    ts = str(ts)

                conn.execute(
                    """INSERT INTO backtest_results
                       (run_id, asset, timeframe, timestamp_utc, signal, position,
                        predicted_return, realized_return, transaction_cost,
                        cumulative_return, model_name)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id or row.get("run_id"),
                        row.get("asset"), row.get("timeframe"),
                        ts, row.get("signal"),
                        float(row.get("position")) if pd.notna(row.get("position")) else None,
                        float(row.get("predicted_return")) if pd.notna(row.get("predicted_return")) else None,
                        float(row.get("realized_return")) if pd.notna(row.get("realized_return")) else None,
                        float(row.get("transaction_cost")) if pd.notna(row.get("transaction_cost")) else None,
                        float(row.get("cumulative_return")) if pd.notna(row.get("cumulative_return")) else None,
                        row.get("model_name"),
                    ),
                )
                count += 1
            conn.commit()
        finally:
            conn.close()
        return count

    def get_backtest_results(self, asset: str, timeframe: str) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM backtest_results WHERE asset=? AND timeframe=? ORDER BY timestamp_utc",
                conn, params=(asset, timeframe),
            )
        finally:
            conn.close()

    # --- Experiment Runs ---
    def save_experiment_run(self, record: dict) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO experiment_runs
                   (run_id, asset, timeframe, data_mode, price_provider,
                    price_count, price_date_range, post_provider, post_count,
                    post_date_range, news_count, best_model, validation_mae,
                    test_mae, naive_test_mae, strategy_return, buyhold_return,
                    strategy_sharpe, max_drawdown, status, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], record["asset"], record["timeframe"],
                    record.get("data_mode", "live"), record.get("price_provider", "binance"),
                    record.get("price_count", 0), record.get("price_date_range", ""),
                    record.get("post_provider", "apify"), record.get("post_count", 0),
                    record.get("post_date_range", ""), record.get("news_count", 0),
                    record.get("best_model"), record.get("validation_mae"),
                    record.get("test_mae"), record.get("naive_test_mae"),
                    record.get("strategy_return"), record.get("buyhold_return"),
                    record.get("strategy_sharpe"), record.get("max_drawdown"),
                    record.get("status", "completed"), record.get("notes", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_experiment_runs(self, asset: str = None) -> pd.DataFrame:
        conn = self._conn()
        try:
            if asset:
                return pd.read_sql_query(
                    "SELECT * FROM experiment_runs WHERE asset=? ORDER BY created_at DESC",
                    conn, params=(asset,),
                )
            return pd.read_sql_query(
                "SELECT * FROM experiment_runs ORDER BY created_at DESC", conn
            )
        finally:
            conn.close()

    # --- Model runs ---
    def save_model_run(self, record: dict) -> int:
        conn = self._conn()
        try:
            cursor = conn.execute(
                """INSERT INTO model_runs
                   (asset, timeframe, model_name, horizon, feature_columns,
                    training_start, training_end, validation_start, validation_end,
                    test_start, test_end, config_json, model_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["asset"], record["timeframe"], record["model_name"],
                    record["horizon"], record.get("feature_columns"),
                    record.get("training_start"), record.get("training_end"),
                    record.get("validation_start"), record.get("validation_end"),
                    record.get("test_start"), record.get("test_end"),
                    record.get("config_json"), record.get("model_path"),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
