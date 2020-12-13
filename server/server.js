const express = require('express');
const app = express();
const api = require('./routes/index');
const cors = require('cors');
const API_PORT = 3002;

app.use(cors())
app.use('/api', api);

const server = app.listen(API_PORT,() =>{
	console.log('Server is running at http://localhost:', API_PORT)
})