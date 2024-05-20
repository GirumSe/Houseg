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
        const response = await fetch('http://your-fastapi-backend-url/locations', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: formData
        });

        if (response.ok) {
            alert('Location added successfully!');
            window.location.href = 'index.html';
        } else {
            const errorData = await response.json();
            alert('Error: ' + errorData.detail);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while adding the location.');
    }
});
