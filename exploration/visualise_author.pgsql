SELECT "by" AS author, AVG("score") AS avg_score
FROM (
    SELECT "by", "score"
    FROM "hacker_news"."items"
    WHERE "type" = 'story' AND "score" IS NOT NULL
) AS limited_items
GROUP BY "by"
ORDER BY avg_score DESC
LIMIT 10;