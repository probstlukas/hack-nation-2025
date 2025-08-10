"""
News-based Sentiment Analysis Module for FinDocGPT
Integrates NewsAPI to fetch financial news and analyze sentiment

This module fetches relevant news articles for companies and performs
sentiment analysis to provide market sentiment insights for forecasting.
"""

import os
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

# Import our existing sentiment analysis
from .text2sentiment import text2sentiment, SentimentSummary

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Represents a single news article"""
    title: str
    description: str
    content: str
    url: str
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

@dataclass
class NewsSentimentResult:
    """Results of news sentiment analysis"""
    company: str
    total_articles: int
    date_range: str
    overall_sentiment: Dict[str, float]
    articles: List[NewsArticle]
    sentiment_trend: List[Dict[str, Any]]
    summary: str

class NewsAPIClient:
    """Client for NewsAPI integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'User-Agent': 'FinDocGPT/1.0'
        })
    
    def fetch_company_news(
        self, 
        company_name: str, 
        days_back: int = 30,
        page_size: int = 50,
        language: str = 'en'
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles related to a specific company
        
        Args:
            company_name: Name of the company to search for
            days_back: Number of days to look back for news
            page_size: Number of articles to fetch (max 100)
            language: Language code for articles
            
        Returns:
            List of news articles
        """
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Prepare search query - include financial terms
        query = f'"{company_name}" AND (earnings OR financial OR stock OR revenue OR profit OR loss OR market)'
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': min(page_size, 100),  # NewsAPI limit
            'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,yahoo.com,wsj.com,ft.com'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"Fetched {len(articles)} articles for {company_name}")
                return articles
            else:
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for {company_name}: {e}")
            return []
    
    def fetch_market_news(
        self, 
        days_back: int = 7,
        page_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch general market/financial news
        
        Args:
            days_back: Number of days to look back
            page_size: Number of articles to fetch
            
        Returns:
            List of news articles
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        params = {
            'category': 'business',
            'country': 'us',
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'pageSize': min(page_size, 100),
            'sortBy': 'publishedAt'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/top-headlines",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market news: {e}")
            return []

class NewsSentimentAnalyzer:
    """Analyzes sentiment of news articles"""
    
    def __init__(self, api_key: str):
        self.news_client = NewsAPIClient(api_key)
    
    def analyze_company_sentiment(
        self, 
        company_name: str, 
        days_back: int = 30
    ) -> NewsSentimentResult:
        """
        Analyze sentiment for a specific company based on recent news
        
        Args:
            company_name: Name of the company
            days_back: Number of days to analyze
            
        Returns:
            NewsSentimentResult with comprehensive sentiment analysis
        """
        # Fetch news articles
        raw_articles = self.news_client.fetch_company_news(
            company_name, 
            days_back=days_back
        )
        
        if not raw_articles:
            return NewsSentimentResult(
                company=company_name,
                total_articles=0,
                date_range=f"Last {days_back} days",
                overall_sentiment={'positive': 0, 'neutral': 1, 'negative': 0},
                articles=[],
                sentiment_trend=[],
                summary="No recent news articles found for sentiment analysis."
            )
        
        # Process articles and analyze sentiment
        processed_articles = []
        all_sentiment_scores = []
        
        for article_data in raw_articles:
            # Skip articles without sufficient content - handle None values
            title = (article_data.get('title') or '').strip()
            description = (article_data.get('description') or '').strip()
            content = (article_data.get('content') or '').strip()
            
            if not title and not description:
                continue
            
            # Combine title and description for sentiment analysis
            text_for_analysis = f"{title}. {description}"
            if content and content != description:
                # Remove common truncation patterns
                content_clean = content.replace('[+chars]', '').replace('…', '').strip()
                if len(content_clean) > len(description):
                    text_for_analysis += f" {content_clean}"
            
            # Perform sentiment analysis
            try:
                sentiment_results = text2sentiment([text_for_analysis])
                if sentiment_results:
                    sentiment = sentiment_results[0]
                    
                    # Convert to our format
                    sentiment_score = sentiment.textblob_polarity
                    sentiment_label = self._get_sentiment_label(sentiment_score)
                    
                    all_sentiment_scores.append(sentiment_score)
                    
                    article = NewsArticle(
                        title=title,
                        description=description,
                        content=content,
                        url=article_data.get('url', ''),
                        published_at=datetime.fromisoformat(
                            article_data.get('publishedAt', '').replace('Z', '+00:00')
                        ),
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label
                    )
                    processed_articles.append(article)
                    
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for article: {e}")
                continue
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(all_sentiment_scores)
        
        # Create sentiment trend (group by day)
        sentiment_trend = self._create_sentiment_trend(processed_articles)
        
        # Generate summary
        summary = self._generate_sentiment_summary(
            company_name, 
            processed_articles, 
            overall_sentiment
        )
        
        return NewsSentimentResult(
            company=company_name,
            total_articles=len(processed_articles),
            date_range=f"Last {days_back} days",
            overall_sentiment=overall_sentiment,
            articles=processed_articles,
            sentiment_trend=sentiment_trend,
            summary=summary
        )
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    def _calculate_overall_sentiment(self, scores: List[float]) -> Dict[str, float]:
        """Calculate overall sentiment distribution"""
        if not scores:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        positive = sum(1 for score in scores if score > 0.1)
        negative = sum(1 for score in scores if score < -0.1)
        neutral = len(scores) - positive - negative
        
        total = len(scores)
        
        return {
            'positive': positive / total,
            'neutral': neutral / total,
            'negative': negative / total
        }
    
    def _create_sentiment_trend(self, articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """Create daily sentiment trend"""
        # Group articles by day
        daily_sentiment = {}
        
        for article in articles:
            date_key = article.published_at.strftime('%Y-%m-%d')
            
            if date_key not in daily_sentiment:
                daily_sentiment[date_key] = []
            
            daily_sentiment[date_key].append(article.sentiment_score)
        
        # Calculate daily averages
        trend = []
        for date, scores in sorted(daily_sentiment.items()):
            if scores:
                avg_sentiment = sum(scores) / len(scores)
                trend.append({
                    'date': date,
                    'sentiment': avg_sentiment,
                    'article_count': len(scores),
                    'label': self._get_sentiment_label(avg_sentiment)
                })
        
        return trend
    
    def _generate_sentiment_summary(
        self, 
        company_name: str, 
        articles: List[NewsArticle], 
        overall_sentiment: Dict[str, float]
    ) -> str:
        """Generate a text summary of sentiment analysis"""
        if not articles:
            return f"No recent news found for {company_name}."
        
        total_articles = len(articles)
        positive_pct = overall_sentiment['positive'] * 100
        negative_pct = overall_sentiment['negative'] * 100
        neutral_pct = overall_sentiment['neutral'] * 100
        
        # Determine overall tone
        if positive_pct > negative_pct + 10:
            tone = "predominantly positive"
        elif negative_pct > positive_pct + 10:
            tone = "predominantly negative"
        else:
            tone = "mixed"
        
        # Find most recent significant sentiment
        recent_articles = sorted(articles, key=lambda x: x.published_at, reverse=True)[:5]
        recent_sentiment = [a.sentiment_score for a in recent_articles if a.sentiment_score is not None]
        
        if recent_sentiment:
            avg_recent = sum(recent_sentiment) / len(recent_sentiment)
            recent_trend = "positive" if avg_recent > 0.1 else "negative" if avg_recent < -0.1 else "neutral"
        else:
            recent_trend = "neutral"
        
        summary = f"""
        Based on analysis of {total_articles} recent news articles, sentiment towards {company_name} is {tone}.
        
        Recent trend appears {recent_trend} based on the latest articles.
        """.strip()
        
        return summary

# Convenience function for easy integration
def analyze_company_news_sentiment(
    company_name: str, 
    api_key: str,
    days_back: int = 30
) -> NewsSentimentResult:
    """
    Convenience function to analyze company sentiment from news
    
    Args:
        company_name: Name of the company to analyze
        api_key: NewsAPI key
        days_back: Number of days to look back
        
    Returns:
        NewsSentimentResult with comprehensive analysis
    """
    analyzer = NewsSentimentAnalyzer(api_key)
    return analyzer.analyze_company_sentiment(company_name, days_back)
