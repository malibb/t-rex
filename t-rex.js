document.addEventListener("DOMContentLoaded", function () {
    const dino = document.getElementById('dino');
    const game = document.getElementById('game');
    const scoreElem = document.getElementById('score');
    let gameOverStatus = false;
    let isJumping = false;
    let obstacleSpeed = 5;
    let gameInterval;
    let obstacleTimeout;
    let score = 0;

    document.addEventListener('keydown', (event) => {
        if (event.code === 'Space' && !isJumping) {
            jump();
        }
    });

    document.addEventListener('touchstart', () => {
        if (!isJumping) {
            jump();
        }
    });

    function jump() {
        isJumping = true;
        let jumpHeight = 0;
        const jumpInterval = setInterval(() => {
            dino.style.bottom = `${jumpHeight}px`;
            jumpHeight += 10;
            if (jumpHeight > 100) {
                clearInterval(jumpInterval);
                fall();
            }
        }, 20);
    }

    function fall() {
        let jumpHeight = 100;
        const fallInterval = setInterval(() => {
            dino.style.bottom = `${jumpHeight}px`;
            jumpHeight -= 10;
            if (jumpHeight < 0) {
                clearInterval(fallInterval);
                isJumping = false;
            }
        }, 20);
    }

    function startGame() {
        spawnObstacle();
        gameInterval = setInterval(() => {
            const obstacles = document.getElementsByClassName('obstacle');
            for (let i = 0; i < obstacles.length; i++) {
                const obstacle = obstacles[i];
                const obstacleLeft = parseInt(obstacle.style.right) || 0;
                if (obstacleLeft > 650) {
                    game.removeChild(obstacle);
                    score++;
                    scoreElem.innerText = `Score: ${score}`;
                } else {
                    obstacle.style.right = `${obstacleLeft + obstacleSpeed}px`;
                }
                checkCollision(obstacle);
            }
            gameLoop();
        }, 20);
    }

    function spawnObstacle() {
        obstacleTimeout = setTimeout(() => {
            const obstacle = document.createElement('div');
            obstacle.className = 'obstacle';
            obstacle.style.height = `${Math.random() * 20 + 10}px`;
            obstacle.style.width = `${Math.random() * 20 + 10}px`;
            obstacle.style.right = '0px';  // Set the initial position to the right edge
            game.appendChild(obstacle);
            spawnObstacle();
        }, Math.random() * 2000 + 1000);
    }

    function checkCollision(obstacle) {
        const dinoRect = dino.getBoundingClientRect();
        const obstacleRect = obstacle.getBoundingClientRect();
        if (dinoRect.right > obstacleRect.left &&
            dinoRect.left < obstacleRect.right &&
            dinoRect.bottom > obstacleRect.top &&
            dinoRect.top < obstacleRect.bottom) {
            gameOver();
        }
    }


    function gameOver() {
        clearInterval(gameInterval);
        clearTimeout(obstacleTimeout);  // Clear the timeout
        console.log(`Game Over! Your score is: ${score}`);
        gameOverStatus = true;  // Set the game over status to true

        // Remove all obstacles from the game
        while (game.childNodes.length > 2) {
            game.removeChild(game.lastChild);
            console.log(game.childNodes.length);
        }

        // Reset the game after 2 seconds
        setTimeout(() => {
            score = 0; // Reset the score
            scoreElem.innerText = `Score: ${score}`;
            gameOverStatus = false; // Reset the game over status
            startGame(); // Restart the game
        }, 2000);
    }



    // Add the following code to your script
    const memorySize = 2000;
    const batchSize = 32;
    const discountFactor = 0.99;
    const epsilonStart = 1.0;
    const epsilonEnd = 0.1;
    const epsilonDecaySteps = 10000;
    let stepCount = 0;
    let memory = [];
    const numStateFeatures = 2; // distance to the next obstacle and height of the obstacle
    const numActions = 2; // jump or do nothing

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [numStateFeatures], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: numActions, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

    function gameLoop() {
        const state = getState();
        const epsilon = getEpsilon(stepCount);
        const action = chooseAction(state, epsilon);
        console.log(`state: ${state} epsilon: ${epsilon} action:${action}`);
        applyAction(action);

        const reward = getReward();
        const nextState = getState();
        const done = isGameOver();
        addToMemory(state, action, reward, nextState, done);

        if (!done) {
            trainModel();
        }

        stepCount++;
    }

    function getState() {
        // Implement your state representation here
        // For example, use the distance to the next obstacle and the height of the obstacle as state features
        const obstacle = document.getElementsByClassName('obstacle')[0];
        if (dino, obstacle) {
            const dinoRect = dino.getBoundingClientRect();
            const obstacleRect = obstacle.getBoundingClientRect();
            const distance = obstacleRect.left - dinoRect.right;
            const height = obstacleRect.height;
            return [distance, height];
        }
        return [0, 0];
    }

    function getReward() {
        // Implement your reward function here
        // For example, use +1 for surviving and -100 for game over
        return isGameOver() ? -100 : +1;
    }

    function isGameOver() {
        return gameOverStatus;  // Return the game over status
    }

    function addToMemory(state, action, reward, nextState, done) {
        memory.push({ state, action, reward, nextState, done });
        if (memory.length > memorySize) {
            memory.shift();
        }
    }
    let isModelTraining = false;

    async function trainModel() {
        if (memory.length < batchSize || isModelTraining) {
            return;
        }

        isModelTraining = true;

        // Sample a batch of experiences from memory
        const batch = [];
        while (batch.length < batchSize) {
            const index = Math.floor(Math.random() * memory.length);
            batch.push(memory[index]);
        }

        // Prepare training data
        const states = batch.map(experience => experience.state);
        const nextStates = batch.map(experience => experience.nextState);
        const qValuesNext = model.predict(tf.tensor(nextStates));
        const targetQValues = model.predict(tf.tensor(states)).arraySync();

        function isIterable(qValuesNext, i) {
            if (Array.isArray(qValuesNext)) {
                return i >= 0 && i < qValuesNext.length && Array.isArray(qValuesNext[i]);
            } else {
                return false;
            }
        }

        // Update target Q-values
        for (let i = 0; i < batch.length; i++) {
            const action = batch[i].action;
            const reward = batch[i].reward;
            const done = batch[i].done;
            let maxQValueNext = 1;
            if (isIterable(qValuesNext, i)) {
                maxQValueNext = Math.max(...qValuesNext[i]);
            }
            targetQValues[i][action] = reward + (done ? 0 : discountFactor * maxQValueNext);
        }

        // Train the Q-network
        try {
            await model.fit(tf.tensor(states), tf.tensor(targetQValues));
        } catch (error) {
            console.error("Training error:", error);
        } finally {
            isModelTraining = false;
        }
    }

    function getEpsilon(stepCount) {
        return Math.max(epsilonEnd, epsilonStart - (epsilonStart - epsilonEnd) * stepCount / epsilonDecaySteps);
    }

    function chooseAction(state, epsilon) {
        if (Math.random() < epsilon) {
            return Math.floor(Math.random() * numActions);
        } else {
            const qValues = model.predict(tf.tensor([state]));
            return tf.argMax(qValues, 1).dataSync()[0];
        }
    }

    function applyAction(action) {
        console.log(action);
        switch (action) {
            case 0:
                // Do nothing
                break;
            case 1:
                // Jump
                if (!isJumping) {
                    jump();
                }
                break;
        }
    }

    startGame();

});
