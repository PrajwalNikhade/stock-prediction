"""Flask application factory."""
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_name: str = 'default') -> Flask:
    """Create and configure the Flask application."""
    from app.config import config_map

    app = Flask(__name__)
    app.config.from_object(config_map.get(config_name, config_map['default']))

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, origins=[app.config['FRONTEND_URL'], 'http://localhost:3000'],
         supports_credentials=True)

    # Import models so they are registered with SQLAlchemy
    from app.models import user, watchlist, alert, prediction_cache, news_cache, recent_search, portfolio  # noqa: F401

    # Register blueprints
    from app.routes.market import market_bp
    from app.routes.stocks import stocks_bp
    from app.routes.prediction import prediction_bp
    from app.routes.news import news_bp
    from app.routes.watchlist import watchlist_bp
    from app.routes.alerts import alerts_bp
    from app.routes.portfolio import portfolio_bp
    from app.routes.analytics import analytics_bp

    app.register_blueprint(market_bp, url_prefix='/api/market')
    app.register_blueprint(stocks_bp, url_prefix='/api/stocks')
    app.register_blueprint(prediction_bp, url_prefix='/api/predict')
    app.register_blueprint(news_bp, url_prefix='/api/news')
    app.register_blueprint(watchlist_bp, url_prefix='/api/watchlist')
    app.register_blueprint(alerts_bp, url_prefix='/api/alerts')
    app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
    app.register_blueprint(analytics_bp, url_prefix='/api/analytics')

    # Health check
    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'service': 'SmartStock AI API'}

    # Create tables on application start
    with app.app_context():
        db.create_all()
        _seed_default_user(app)

    return app


def _seed_default_user(app: Flask) -> None:
    """Create default demo user if not exists."""
    from app.models.user import User
    if not User.query.filter_by(username='demo').first():
        user = User(
            username='demo',
            email='demo@smartstock.ai',
            password_hash='pbkdf2:sha256:demo'
        )
        db.session.add(user)
        db.session.commit()
