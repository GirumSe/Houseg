CREATE TABLE `locations`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `latitiude` BIGINT NOT NULL,
    `longitude` BIGINT NOT NULL,
    `image_name` VARCHAR(255) NOT NULL,
    `address` VARCHAR(255) NOT NULL,
    `house_count` BIGINT NOT NULL,
    `username` VARCHAR(255) NOT NULL,
    INDEX `username_idx` (`username`)
);

CREATE TABLE `users`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(255) NOT NULL,
    `email` VARCHAR(255) NOT NULL,
    `password` VARCHAR(255) NOT NULL,
    CONSTRAINT `users_username_foreign` FOREIGN KEY(`username`) REFERENCES `locations`(`username`)
);
