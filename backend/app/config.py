"""Flask application configuration."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-me')
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL', 'sqlite:///smartstock.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True,
    }

    # CORS
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

    # Cache TTL
    PREDICTION_CACHE_TTL = int(os.getenv('PREDICTION_CACHE_TTL', 24))
    NEWS_CACHE_TTL = int(os.getenv('NEWS_CACHE_TTL', 6))


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
