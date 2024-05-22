const navbar = document.getElementById('navbar');

const token = localStorage.getItem('token');
const names = localStorage.getItem('name');
console.log(token)
console.log(names)
if (token && names) {
    // User is logged in
    navbar.innerHTML = `
        <a>${names}</a>
        <a href="#" id="logout">Logout</a>
    `;

    document.getElementById('logout').addEventListener('click', () => {
        // Remove the token and names from local storage
        localStorage.removeItem('token');
        localStorage.removeItem('names');

        // Redirect the user to the login page
        window.location.href = 'login.html';
    });
} else {
    // User is not logged in
    navbar.innerHTML = `
        <a href="login.html">Login</a>
        <a href="register.html">Register</a>
    `;
}

document.getElementById('location-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const latitude = document.getElementById('latitude').value;
    const longitude = document.getElementById('longitude').value;
    const image = document.getElementById('image').files[0];

    const formData = new FormData();
    formData.append('latitude', latitude);
    formData.append('longitude', longitude);
    formData.append('image', image);
    try {
        const response = await fetch('http://127.0.0.1:8000/locations', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: formData
        });

        if (response.ok) {
            alert('Location added successfully!');
        } else {
            const errorData = await response.json();
            alert('Error: ' + errorData.detail);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while adding the location.');
    }
});


