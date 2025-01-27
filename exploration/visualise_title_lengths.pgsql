SELECT title

FROM hacker_news.items 

WHERE type = 'story'
AND title IS NOT NULL