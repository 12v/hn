SELECT lower(regexp_replace(title, '[^a-zA-Z0-9 ]', '', 'g')) AS clean_title

FROM hacker_news.items 

WHERE type = 'story'
AND title IS NOT NULL