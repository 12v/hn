SELECT title, by, time, url, score FROM hacker_news.items 

WHERE type = 'story'
AND title IS NOT NULL
AND dead IS NOT TRUE;