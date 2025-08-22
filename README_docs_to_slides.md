# Google Docs to Slides Generator

A Flask web application that converts Google Docs into beautiful presentation slides with support for multiple export formats (PowerPoint, HTML slideshow).

## Features

🔄 **Smart Content Parsing** - Automatically detects headings and converts them to slide titles  
📊 **Multiple Export Formats** - PowerPoint (.pptx) and HTML slideshow  
🔒 **Secure Authentication** - Google OAuth2 integration  
🎨 **Beautiful Design** - Clean, modern web interface  
⚡ **Fast Processing** - Efficient document parsing and slide generation  

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_docs_to_slides.txt
```

### 2. Run the Application

```bash
python docs_to_slides.py
```

The application will be available at `http://localhost:5000`

### 3. Complete Setup (Automated)

The app includes an **automated setup wizard** to guide you through Google API configuration:

1. **Open the app** at `http://localhost:5000`
2. **Click "Complete Setup"** - you'll see a red banner if setup is needed
3. **Follow the step-by-step guide** at `/setup` which includes:
   - Direct links to Google Cloud Console
   - Automated credential validation
   - File upload with drag & drop
   - Manual credential entry option
   - Real-time setup status checking

### 4. Alternative: Manual Setup

If you prefer manual setup:

1. **Create a Google Cloud Project**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Required APIs**:
   - Google Docs API
   - Google Drive API
   - Google Slides API (optional, for future features)

3. **Create OAuth2 Credentials**:
   - Go to "Credentials" in the Google Cloud Console
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add `http://localhost:5000/callback` to authorized redirect URIs
   - Download the credentials file as `credentials.json`

4. **Place Credentials File**:
   ```bash
   # Place the downloaded file in the project root
   mv ~/Downloads/credentials.json ./credentials.json
   ```

### 5. Environment Variables (Optional)

Create a `.env` file or set environment variables:

```bash
export FLASK_SECRET_KEY="your-secret-key-here"
export FLASK_ENV="development"  # for development
```

## How to Use

### Preparing Your Google Doc

1. **Use Headings**: Structure your document with proper headings (Heading 1, Heading 2, etc.)
   - Each heading becomes a slide title
   - Content under headings becomes slide bullet points

2. **Document Structure Example**:
   ```
   # Main Title (becomes title slide)
   
   ## Introduction (becomes slide 1)
   This is the introduction content.
   Key points about the topic.
   
   ## Key Features (becomes slide 2)
   Feature 1 description
   Feature 2 description
   Feature 3 description
   
   ## Conclusion (becomes slide 3)
   Summary of main points
   Call to action
   ```

3. **Sharing Settings**: Ensure your document is:
   - Shared with the Google account you'll use to authenticate
   - Or set to "Anyone with the link can view"

### Converting Your Document

1. **Authenticate**: Click "Sign in with Google" and authorize the application
2. **Paste URL**: Copy your Google Docs URL and paste it in the converter
3. **Choose Format**: Select PowerPoint (.pptx) or HTML slideshow
4. **Convert**: Click "Convert to Slides" and wait for processing
5. **Download**: Download your generated presentation

## Export Formats

### PowerPoint (.pptx)
- Fully editable PowerPoint presentation
- Professional slide layouts
- Compatible with Microsoft PowerPoint and Google Slides
- Perfect for further customization

### HTML Slideshow
- Interactive web-based presentation
- Keyboard navigation (arrow keys)
- Mobile-responsive design
- Beautiful gradient themes
- No additional software required

## File Structure

```
docs_to_slides/
├── docs_to_slides.py           # Main Flask application
├── templates/
│   └── docs_to_slides.html     # Web interface template
├── exports/                    # Generated presentations (auto-created)
├── credentials.json            # Google OAuth credentials (you provide)
├── requirements_docs_to_slides.txt  # Python dependencies
└── README_docs_to_slides.md    # This file
```

## API Endpoints

- `GET /` - Main application interface
- `GET /auth` - Initiate Google OAuth flow
- `GET /callback` - Handle OAuth callback
- `POST /convert` - Convert document to slides
- `GET /download/<filename>` - Download generated file

## Technical Details

### Document Parsing
- Extracts document structure using Google Docs API
- Identifies headings based on paragraph styles
- Converts content into slide-friendly format
- Preserves text formatting where possible

### Slide Generation
- **PowerPoint**: Uses `python-pptx` library for .pptx creation
- **HTML**: Generates responsive slideshow with CSS3 animations
- Automatic slide layouts and professional styling

### Security
- OAuth2 flow for secure Google authentication
- Session-based credential storage
- No permanent storage of user documents
- Temporary file cleanup

## Troubleshooting

### Common Issues

1. **"Credentials file not found"**
   - Ensure `credentials.json` is in the project root
   - Verify the file is valid JSON from Google Cloud Console

2. **"Document not accessible"**
   - Check document sharing permissions
   - Ensure you're signed in with the correct Google account
   - Verify the document URL is correct

3. **"OAuth error"**
   - Check redirect URI in Google Cloud Console
   - Ensure all required APIs are enabled
   - Verify credentials file matches your project

4. **"Empty slides generated"**
   - Use proper heading styles in your Google Doc
   - Ensure there's content under each heading
   - Check document structure

### Development Mode

For development, you can enable Flask debug mode:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python docs_to_slides.py
```

## Limitations

- Document must be accessible via Google Docs API
- Heading styles are required for proper slide structure
- Images in documents are not currently supported
- Complex formatting may be simplified in conversion
- Rate limits apply based on Google API quotas

## Future Enhancements

- Image extraction and inclusion
- Custom slide templates
- Batch document processing
- Google Slides direct export
- Advanced formatting preservation
- Collaborative features

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your Google API setup
3. Ensure document permissions are correct
4. Check application logs for detailed error messages

---

**Built with ❤️ using Flask, Google APIs, and python-pptx**