SELECT "by" AS author, AVG("score") AS avg_score
FROM "hacker_news"."items"
WHERE "type" = 'story' AND "score" IS NOT NULL
GROUP BY "by"
ORDER BY avg_score DESC
LIMIT 50;