function registerUser() {
    const username = document.getElementById("username").value;
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm-password").value;

    if (password !== confirmPassword) {
        alert("Passwords do not match");
        return;
    }
    console.log(username)
    fetch("http://127.0.0.1:8000/register", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({username, password, email})
    })
    .then(response => {
        if (response.ok) {
            alert("You have registered successfully");
            window.location.href = "../src/login.html";
        } else {
            throw new Error("Registration failed");
        }
    })
    .catch(error => {
        alert(error.message);
    });
}
