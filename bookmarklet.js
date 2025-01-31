javascript: (function () {
    // Helper function to make the POST request
    function makePostRequest(story) {
        return fetch('https://snowy-water-a81c.12v.workers.dev/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(story)
        })
            .then(response => response.json())
            .then(data => {
                const predictedScore = data.score;
                return predictedScore;
            })
            .catch(error => console.error('Error:', error));
    }

    // Iterate over all story rows on the page
    const stories = document.querySelectorAll('.athing');
    stories.forEach(story => {
        let title;
        let author;
        let datetime;
        let url;


        title = story.querySelector('.titleline a').innerText;
        if (!story.nextElementSibling.innerText.includes('comment')) {
            return;
        }

        author = story.nextElementSibling.querySelector('.hnuser').innerText;
        datetime = story.nextElementSibling.querySelector('.age').title;
        url = story.querySelector('.titleline a').href;

        const obj = {
            title: title,
            author: author,
            datetime: datetime,
            url: url
        }

        console.log(obj);

        makePostRequest(obj)
            .then(score => {
                console.log(score);

                const subline = story.nextElementSibling.querySelector('.subline')

                const predictedScoreText = document.createTextNode(' | predicted_score: ' + score);

                // Append the text node to the subline element
                subline.appendChild(predictedScoreText);
            });
    });
})();
