"""News service with sentiment analysis."""
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
from app import db
from app.models.news_cache import NewsCache


class NewsService:
    """Service for fetching news and analyzing sentiment."""

    GOOGLE_NEWS_RSS = 'https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en'

    @staticmethod
    def fetch_news(query: str, limit: int = 15) -> list[dict]:
        """
        Fetch news from Google News RSS feed.

        Args:
            query: Search query (stock symbol or company name)
            limit: Maximum number of articles

        Returns:
            List of news articles with sentiment
        """
        try:
            url = NewsService.GOOGLE_NEWS_RSS.format(query=query.replace(' ', '+'))
            feed = feedparser.parse(url)

            articles = []
            for entry in feed.entries[:limit]:
                title = entry.get('title', '')
                source = entry.get('source', {}).get('title', 'Unknown')
                link = entry.get('link', '')

                # Parse published date
                published = None
                if 'published_parsed' in entry and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6])
                    except Exception:
                        published = datetime.utcnow()

                # Sentiment analysis using TextBlob
                sentiment = NewsService._analyze_sentiment(title)

                articles.append({
                    'title': title,
                    'url': link,
                    'source': source,
                    'published_at': published.isoformat() if published else None,
                    'sentiment_score': sentiment['score'],
                    'sentiment_label': sentiment['label'],
                })

            return articles

        except Exception as e:
            return [{'error': str(e)}]

    @staticmethod
    def get_stock_news(symbol: str, company_name: str = '',
                       limit: int = 15) -> dict:
        """Get news for a specific stock with sentiment summary."""
        # Try fetching with symbol first, then company name
        query = f"{symbol} NSE stock"
        articles = NewsService.fetch_news(query, limit)

        if not articles or 'error' in articles[0]:
            if company_name:
                articles = NewsService.fetch_news(company_name, limit)

        # Calculate sentiment summary
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_score = 0

        valid_articles = [a for a in articles if 'error' not in a]
        for article in valid_articles:
            label = article.get('sentiment_label', 'neutral')
            sentiment_counts[label] += 1
            total_score += article.get('sentiment_score', 0)

        avg_sentiment = total_score / len(valid_articles) if valid_articles else 0

        # Cache articles
        NewsService._cache_articles(symbol, valid_articles)

        return {
            'symbol': symbol.upper(),
            'articles': valid_articles,
            'sentiment_summary': {
                'average_score': round(avg_sentiment, 4),
                'overall_label': NewsService._score_to_label(avg_sentiment),
                'counts': sentiment_counts,
                'total': len(valid_articles),
            }
        }

    @staticmethod
    def get_market_news(limit: int = 20) -> list[dict]:
        """Get general Indian stock market news."""
        queries = [
            'Indian stock market today',
            'Nifty 50 Sensex',
            'NSE BSE India',
        ]

        all_articles = []
        for query in queries:
            articles = NewsService.fetch_news(query, limit=limit // len(queries))
            all_articles.extend(articles)

        # Remove duplicates by title
        seen_titles = set()
        unique = []
        for article in all_articles:
            if 'error' not in article and article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique.append(article)

        return unique[:limit]

    @staticmethod
    def get_sentiment_score_for_prediction(symbol: str) -> float:
        """Get average sentiment score to use as ML feature."""
        news_data = NewsService.get_stock_news(symbol, limit=10)
        return news_data['sentiment_summary']['average_score']

    @staticmethod
    def _analyze_sentiment(text: str) -> dict:
        """Analyze text sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            label = NewsService._score_to_label(polarity)

            return {
                'score': round(polarity, 4),
                'label': label,
                'subjectivity': round(blob.sentiment.subjectivity, 4),
            }
        except Exception:
            return {'score': 0.0, 'label': 'neutral', 'subjectivity': 0.0}

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        return 'neutral'

    @staticmethod
    def _cache_articles(symbol: str, articles: list[dict]) -> None:
        """Cache news articles to database."""
        try:
            # Remove old cache (older than 6 hours)
            cutoff = datetime.utcnow() - timedelta(hours=6)
            NewsCache.query.filter(
                NewsCache.symbol == symbol.upper(),
                NewsCache.cached_at < cutoff
            ).delete()

            for article in articles:
                existing = NewsCache.query.filter_by(
                    symbol=symbol.upper(),
                    title=article['title']
                ).first()

                if not existing:
                    entry = NewsCache(
                        symbol=symbol.upper(),
                        title=article['title'],
                        url=article.get('url', ''),
                        source=article.get('source', ''),
                        sentiment_score=article.get('sentiment_score', 0),
                        sentiment_label=article.get('sentiment_label', 'neutral'),
                        published_at=datetime.fromisoformat(article['published_at']) if article.get('published_at') else None,
                    )
                    db.session.add(entry)

            db.session.commit()
        except Exception:
            db.session.rollback()
