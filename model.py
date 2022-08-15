from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance as td
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from pdfminer.high_level import extract_text

# def RankCandidate(Resume , Job_Desc):
#     text = [Resume,Job_Desc]
#     cv = CountVectorizer()
#     count_matrix = cv.fit_transform(text)
#     cosine_similarity(count_matrix)
#     matchpercentage = cosine_similarity(count_matrix)[0][1]
#     matchpercentage = round(matchpercentage*100,2)
#     return print('Candidate Resume match {}% '.format(matchpercentage) + 'your Job Requirements')





#tokenizer
def do_tfidf(token):
    tfidf = TfidfVectorizer(max_df=1.0, min_df=1)
    words = tfidf.fit_transform(token)
    sentence = " ".join(tfidf.get_feature_names())
    return sentence




#different similarity algorithms
def match(resume, job_des):
    j = td.jaccard.similarity(resume, job_des)
    s = td.sorensen_dice.similarity(resume, job_des)
    c = td.cosine.similarity(resume, job_des)
    o = td.overlap.normalized_similarity(resume, job_des)
    total = (j+s+c+o)/4
    match_percentage = round(total*100.2)
    # total = (s+o)/2
    return "Candidate's Profile match {}%" .format(match_percentage) +" of your job requirements"


    
resume = do_tfidf(["First LastPython DeveloperCertified Python Developer offering 6 years of extensive experience and exceptionalanalytical and critical thinking skills. Delivers a proactive approach, great work ethic,and the ability to function well in fast-paced/deadline-driven team environments.San Francisco, CA 12345+1 234 567-890first.last@resumeworded.comlinkedin.com/in/resumewordedEXPERIENCEGrowthsi, New York, NYCertified Python DeveloperJanuary 2020 - Present● Developed web application back end components while communicatingwith 30+ clients to identify their needs/goals and work on meeting them.● Worked on the improvement of data protection and security, increasingsecurity rate by 24 % while creating new user information solutions.● Maintained large databases and configured services to reduce softwaremaintenance expenses, decreasing the costs by 15% within just one year.● Trained and supervised 3 employees, providing training support/guidance.● Obtained the Employee of the Year Award for meeting and exceeding allassigned goals and objectives and contributing to 33% overall success.Growthsi, New York, NYPython Developer/TesterJuly 2016 - December 2019● Successfully automated the moving of 5 tests from production to stagingand staging to production by carefully reading the keywords from NVbugs.● Consumed APIs while utilizing Python requests to read numerous JSONreports and file automatic bugs in the NVBugs for intermittent tests.● Designed and configured database and backend applications and programs,contributing to operations continuity and increasing efficiency by 14%.● Developed, tested, and debugged software tools utilized by 100+ clients andinternal customers to facilitate and easier process and user experience.● Obtained adequate experience in reviewing Python code for running thetroubleshooting test-cases and bug issues, acquiring all necessary skills.Resume Worded, Boston, MAPython Developer InternJanuary 2015 - June 2016● Designed robust, scalable, secure, and globalized web-based applications toensure the continuity of all business processes and client satisfaction.● Used the Python language to develop 3 web-based data retrieval systems.● Performed data entry and other clerical work for project completion.● Conducted descriptive and multivariate statistical analysis of data usingMatlab, gaining 100% accuracy rate in terms of interpretation and analysis.SKILLSPythonHTML, JavaScriptSeleniumTestCompleteAppiumFrameworks DjangoFlaskPyramidPyJamasJythonAngular JS; Node JSSpringWebLogicWebSphereJBoss Amazon EC2Jenkins And Fabric J2EE JDBCJNDIJSP And Servlets DatabasesOracle MySQLEDUCATIONResume WordedUniversityBS Computer ScienceNovember 2014Boston, MACERTIFICATIONSResume WordedUniversityCertified Python DeveloperMay 2016Boston, MA"])
job_desc= do_tfidf(["Backend Developer - Python/DjangoWe are looking for Python/Django developers with 2-4 + years of experience to be part of our growing team. The role is of Python/Django Backend Web Developer who can handle the challenging work of Web applications development using Python / Django framework.Required Skills:•	Strong working knowledge of constructs of HTTP, RESTful APIs and WebSockets.•	Build and maintain highly scalable Python processes for the purpose of data collection, manipulation, data pruning, trending and analytics, etc.•	Strong understanding of Python and good knowledge of various Python Libraries (NumPy, Pandas, ORM libraries etc.), API's and toolkits•	Able to integrate multiple data sources and databases into one system•	Working knowledge of web templating engine for python such as Jinja2 etc•	Ability to integrate user-facing elements developed by front-end developers with server-side logic.•	Ability to integrate data storage solutions and caches.•	Understanding of the threading limitations of Python, and multi-process architecture•	Understanding of fundamental design principles behind a scalable application•	Good understanding of event-driven programming in Python•	Good knowledge of PostGRESQL•	Able to create database schemas that represent and support business processes•	Strong Analytical, Problem Solving & Innovative Skills.•	Working knowledge of version control tools (GIT/SVN).•	Experience/knowledge of working on the LINUX environment.•	Attention to detail, problem-solver with strong analytical skill, good sense of urgency•	Ability to carry out tasks without supervision and able to meet tight deadlines•	Good Writing & Communication Skills. The candidate should be able to communicate with potential Clients and gather inputs/change requests.•	Work with a performance oriented team driven by ownership and open to experiments•	Strong interpersonal skills with excellent written and verbal communicationResponsibilities:•	Develop back-end components to improve responsiveness and overall performance•	Designing and implementing system architecture to handle scale•	Coordinate with Frontend Dev team & Creating APIs (w/ DjangoRestFramework)•	Integrating of web pages with Backend using templating engines for Python•	Build bulletproof API integrations with third party APIs for various use casesGood to have (Additional Skills):•	Relevant full stack experience.•	You've worked with core AWS services in the past and have experience with EC2, ELB, AutoScaling, CloudFront, S3•	Demonstrable work/project portfolio online.•	Working knowledge of front end technologies like Javascript, JQuery, HTML, and CSS•	A good sense of design and aesthetics.•	Experience/Zeal to work in a fast-paced startup environment.•	Candidates who can join immediately will be preferredDevelop very high sense of ownership, the zeal to build scalable applicationsJob Type: Full-timeSalary: ₹35,000.00 - ₹55,000.00 per monthExperience:•	Django Development: 1 year (Preferred)•	total work: 2 years (Preferred)Education:•	Bachelor's (Preferred)Work Remotely:•	Temporarily due to COVID-19"])
match(resume,job_desc)





