const form = document.getElementById('login-form');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const response = await fetch('http://127.0.0.1:8000/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
    });

    if (response.ok) {
        const { access_token, name } = await response.json();

        // Store the token and name in local storage
        localStorage.setItem('token', access_token);
        localStorage.setItem('name', name);
        //Redirect the user to the landing page
        window.location.href = '../src/index.html';
    } else {
        // Handle the error, e.g. display an error message
        alert('Login failed');
    }
});
