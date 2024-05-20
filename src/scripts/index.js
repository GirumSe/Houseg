const navbar = document.getElementById('navbar');
const greeting = document.getElementById('greeting');

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
    
    greeting.textContent = `Hello, ${names}!`;

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
