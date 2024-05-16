const form = document.getElementById('login-form');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const response = await fetch('YOUR_BACKEND_API_URL', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
    });

    if (response.ok) {
        const { token } = await response.json();

        // Store the token in local storage or a cookie
        localStorage.setItem('token', token);

        // Redirect the user to the landing page
        window.location.href = '../index.html';
    } else {
        // Handle the error, e.g. display an error message
        alert('Login failed');
    }
});
