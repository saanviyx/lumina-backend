const express = require("express");
const cors = require("cors");
const { connectToDatabase } = require("./config/database");
const authRoutes = require("./routes/auth");
const analysisRoutes = require("./routes/analysis");
const progressRoutes = require("./routes/progress");
const mlRoutes = require('./routes/ml');
const path = require('path');

const app = express();
const { PORT } = require("./config/dotenv");

const corsOptions = {
  origin: "http://127.0.0.1:5500", // Allow only the Live Server frontend
  methods: ["GET", "POST", "PUT", "DELETE"], // Allowed methods
  credentials: true, // If you're sending cookies or credentials
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json({ limit: "500mb" })); // Increase JSON payload limit
app.use(express.urlencoded({ limit: "500mb", extended: true })); // Increase URL-encoded data limit

// Serve static files
app.use(express.static(path.join(__dirname, '..', 'frontend', 'public')));
app.use("/js", express.static(path.join(__dirname, "..", "frontend", "js")));

// Catch-all route to serve index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'public', 'index.html'));
});

// Routes
app.use("/authentication", authRoutes);
app.use("/analysis", analysisRoutes);
app.use("/progress", progressRoutes);
app.use("/ml", mlRoutes);

// Database connection and server start
connectToDatabase().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
