const express = require("express");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("../config/dotenv");
const { getUsersCollection } = require("../models/user");

const router = express.Router();

router.post("/register", async (req, res) => {
  try {
    const { Name, username, email, gender, password } = req.body;

    if (!email || !username || !password) {
      return res.status(400).json({ message: "Please provide email, username, and password" });
    }

    const usersCollection = getUsersCollection();

    const existingUser = await usersCollection.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ message: "Email or username already in use" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = {
      Name,
      email,
      username,
      gender,
      password: hashedPassword,
      analysisCount: 0,
      createdAt: new Date(),
    };

    await usersCollection.insertOne(newUser);

    const token = jwt.sign({ id: newUser._id, username: newUser.username, email: newUser.email }, JWT_SECRET, { expiresIn: "24h" });

    res.status(201).json({ message: "User registered successfully", token, user: { username: newUser.username, email: newUser.email, analysisCount: newUser.analysisCount } });
  } catch (error) {
    console.error("Registration error:", error);
    res.status(500).json({ message: "Server error during registration" });
  }
});

router.post("/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ message: "Please provide username and password" });
    }

    const usersCollection = getUsersCollection();
    const user = await usersCollection.findOne({ username });

    if (!user) {
      return res.status(400).json({ message: "Invalid username or password" });
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ message: "Invalid username or password" });
    }

    const token = jwt.sign({ id: user._id, username: user.username, email: user.email }, JWT_SECRET, { expiresIn: "24h" });

    res.status(200).json({ message: "Login successful", token, user: { username: user.username, email: user.email, analysisCount: user.analysisCount } });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error during login" });
  }
});

module.exports = router;