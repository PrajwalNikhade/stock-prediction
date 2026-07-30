-- SmartStock AI Database Schema
-- Compatible with MySQL 8.0+ / SQLite 3

CREATE DATABASE IF NOT EXISTS smartstock;
USE smartstock;

-- ==========================================
-- Users Table
-- ==========================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    clerk_id VARCHAR(100) DEFAULT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_username (username),
    INDEX idx_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- Watchlist Table
-- ==========================================
CREATE TABLE IF NOT EXISTS watchlist (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    company_name VARCHAR(200) NOT NULL,
    is_pinned BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uq_user_symbol (user_id, symbol),
    INDEX idx_watchlist_user (user_id),
    INDEX idx_watchlist_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- Portfolio Table
-- ==========================================
CREATE TABLE IF NOT EXISTS portfolios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    company_name VARCHAR(200) NOT NULL,
    quantity INT NOT NULL,
    buy_price DECIMAL(15, 4) NOT NULL,
    bought_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_portfolio_user (user_id),
    INDEX idx_portfolio_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- Alerts Table
-- ==========================================
CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    alert_type ENUM('price_above', 'price_below', 'rsi_above', 'rsi_below') NOT NULL,
    target_value DECIMAL(15, 4) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_triggered BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    triggered_at DATETIME NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_alerts_user (user_id),
    INDEX idx_alerts_symbol (symbol),
    INDEX idx_alerts_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- Predictions Cache Table
-- ==========================================
CREATE TABLE IF NOT EXISTS predictions_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    prediction_date DATE NOT NULL,
    current_price DECIMAL(15, 4),
    pred_1d DECIMAL(15, 4),
    pred_7d DECIMAL(15, 4),
    pred_30d DECIMAL(15, 4),
    confidence DECIMAL(5, 4),
    feature_importance JSON,
    metrics JSON,
    recommendation VARCHAR(10),
    recommendation_reasons JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ttl_hours INT DEFAULT 24,
    INDEX idx_pred_symbol (symbol),
    INDEX idx_pred_date (prediction_date),
    UNIQUE KEY uq_pred_symbol_date (symbol, prediction_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- News Cache Table
-- ==========================================
CREATE TABLE IF NOT EXISTS news_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) DEFAULT NULL,
    title VARCHAR(500) NOT NULL,
    url VARCHAR(1000) NOT NULL,
    source VARCHAR(200),
    sentiment_score DECIMAL(5, 4) DEFAULT 0.0,
    sentiment_label ENUM('positive', 'neutral', 'negative') DEFAULT 'neutral',
    published_at DATETIME,
    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_news_symbol (symbol),
    INDEX idx_news_cached (cached_at),
    INDEX idx_news_sentiment (sentiment_label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==========================================
-- Recent Searches Table
-- ==========================================
CREATE TABLE IF NOT EXISTS recent_searches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    company_name VARCHAR(200),
    searched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_searches_user (user_id),
    INDEX idx_searches_time (searched_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed default user
INSERT INTO users (username, email, password_hash)
VALUES ('demo', 'demo@smartstock.ai', 'pbkdf2:sha256:demo_hash_placeholder')
ON DUPLICATE KEY UPDATE username = username;
