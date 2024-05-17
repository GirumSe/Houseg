function registerUser() {
    const username = document.getElementById("username").value;
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm-password").value;

    if (password !== confirmPassword) {
        alert("Passwords do not match");
        return;
    }

    const data = {
        username: username,
        email: email,
        password: password
    };

    fetch("http://localhost:8000/register", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (response.ok) {
            alert("You have registered successfully");
            window.location.href = "../login.html";
        } else {
            throw new Error("Registration failed");
        }
    })
    .catch(error => {
        alert(error.message);
    });
}
