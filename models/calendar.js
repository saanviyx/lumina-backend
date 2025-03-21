const { getDb } = require("../config/database");

function getCalendarCollection() {
  return getDb().collection("Calendar"); // Renamed collection to avoid conflicts
}

async function getUserCalendarEntries(username) {
  const calendarCollection = getCalendarCollection();
  return await calendarCollection.find({ username }).toArray();
}

async function markDayCompleted(username, date) {
  const calendarCollection = getCalendarCollection();
  const dateObj = new Date(date);
  // Format date as YYYY-MM-DD to ensure consistency
  const formattedDate = dateObj.toISOString().split('T')[0];
  
  const result = await calendarCollection.updateOne(
    { username, date: formattedDate },
    { $set: { username, date: formattedDate, completed: true, timestamp: new Date() } },
    { upsert: true }
  );
  
  return result;
}

async function markDayUncompleted(username, date) {
  const calendarCollection = getCalendarCollection();
  const dateObj = new Date(date);
  const formattedDate = dateObj.toISOString().split('T')[0];
  
  const result = await calendarCollection.updateOne(
    { username, date: formattedDate },
    { $set: { username, date: formattedDate, completed: false, timestamp: new Date() } },
    { upsert: true }
  );
  
  return result;
}

async function getStreakStats(username) {
  const entries = await getUserCalendarEntries(username);
  
  // Handle case with no entries
  if (!entries || entries.length === 0) {
    return {
      currentStreak: 0,
      longestStreak: 0,
      monthlyCompletion: 0
    };
  }
  
  // Sort entries by date
  entries.sort((a, b) => new Date(a.date) - new Date(b.date));
  
  // Get completed entries
  const completedEntries = entries.filter(entry => entry.completed);
  
  // Calculate current streak
  let currentStreak = 0;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  // Convert dates to YYYY-MM-DD format for comparison
  const todayFormatted = today.toISOString().split('T')[0];
  
  // Check if entries exist and are completed for consecutive days up to today or yesterday
  for (let i = 1; i <= 1000; i++) { // Limit to avoid infinite loop
    const checkDate = new Date();
    checkDate.setDate(today.getDate() - i + 1);
    const checkDateFormatted = checkDate.toISOString().split('T')[0];
    
    const entry = entries.find(e => e.date === checkDateFormatted);
    
    if (entry && entry.completed) {
      currentStreak++;
    } else {
      break;
    }
  }
  
  // Calculate longest streak
  let longestStreak = 0;
  let tempStreak = 0;
  
  for (let i = 0; i < completedEntries.length; i++) {
    if (i === 0) {
      tempStreak = 1;
    } else {
      const currentDate = new Date(completedEntries[i].date);
      const prevDate = new Date(completedEntries[i-1].date);
      
      // Check if dates are consecutive
      const diffTime = Math.abs(currentDate - prevDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      
      if (diffDays === 1) {
        tempStreak++;
      } else {
        tempStreak = 1;
      }
    }
    
    if (tempStreak > longestStreak) {
      longestStreak = tempStreak;
    }
  }
  
  // Calculate monthly completion percentage
  const currentMonth = today.getMonth();
  const currentYear = today.getFullYear();
  
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const startOfMonth = new Date(currentYear, currentMonth, 1);
  
  // Filter entries for current month
  const monthEntries = entries.filter(entry => {
    const entryDate = new Date(entry.date);
    return entryDate.getMonth() === currentMonth && entryDate.getFullYear() === currentYear;
  });
  
  const completedDaysThisMonth = monthEntries.filter(entry => entry.completed).length;
  
  // Calculate days passed in current month
  const daysPassed = Math.min(today.getDate(), daysInMonth);
  
  const monthlyCompletion = daysPassed > 0 
    ? Math.round((completedDaysThisMonth / daysPassed) * 100) 
    : 0;
  
  return {
    currentStreak,
    longestStreak,
    monthlyCompletion
  };
}

// New function to initialize calendar if empty
async function initializeUserCalendar(username) {
  const entries = await getUserCalendarEntries(username);
  
  // If user already has entries, don't initialize
  if (entries && entries.length > 0) {
    return { message: "Calendar already initialized" };
  }
  
  // Create default empty structure
  const calendarCollection = getCalendarCollection();
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const formattedToday = today.toISOString().split('T')[0];
  
  // Initialize with just today's date as not completed
  await calendarCollection.insertOne({
    username,
    date: formattedToday,
    completed: false,
    timestamp: new Date()
  });
  
  return { message: "Calendar initialized successfully" };
}

module.exports = {
  getCalendarCollection,
  getUserCalendarEntries,
  markDayCompleted,
  markDayUncompleted,
  getStreakStats,
  initializeUserCalendar
};