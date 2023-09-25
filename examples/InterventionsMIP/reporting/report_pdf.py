import os
import base64
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from InterventionsMIP import reporting_path


def fill_template(data, template_file, report_file):
    
    with open(template_file, 'r') as file:
        tex_template = file.read()
    
    for k, v in data.items():
        tex_template = tex_template.replace(k, str(v), 1)
    
    tex_template = tex_template.replace('INSERT-TRIGGER-DESCRIPTION', trigger_summary(data['policy']))
    
    with open(report_file, 'w') as out_rep:
        out_rep.write(tex_template)


def trigger_summary(trigger_values):
    
    # trigger_srt = f"Safety threshold: {trigger_values['hosp_level_release']:.0f} \n\n"
    # trigger_srt += f"Lock-down trigger: \n "
    # trigger_srt += f"\\begin{'{itemize}'} \\item Threshold 1: {trigger_values['hosp_rate_threshold_1']}\n"
    # if trigger_values['hosp_rate_threshold_2'] > 0:
    #     trigger_srt += f"\\item Threshold 2: {trigger_values['hosp_rate_threshold_2']}\n"
    # trigger_srt += f"\\end{'{itemize}'} \n\n"
    trigger_srt = f'{trigger_values}'
    trigger_srt = trigger_srt.replace('_', '/')
    return trigger_srt


def send_report(instance_name, toaddr, report_file):
    fromaddr = base64.b64decode("Y292aWQubnVAZ21haWwuY29t").decode("utf-8")
    
    # instance of MIMEMultipart
    msg = MIMEMultipart()
    
    # storing the senders email address
    msg['From'] = fromaddr
    
    # storing the receivers email address
    msg['To'] = toaddr
    
    # storing the subject
    msg['Subject'] = f"[Not reply] Automated report - {instance_name}"
    
    # string to store the body of the mail
    body = "Attached is an automated report. \n"
    
    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))
    
    # open the file to be sent
    attachment = open(report_file, "rb")
    
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    
    # To change the payload into encoded form
    p.set_payload((attachment).read())
    
    # encode into base64
    encoders.encode_base64(p)
    
    p.add_header('Content-Disposition', f"attachment; filename= {instance_name}.pdf")
    
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)
    
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    
    # start TLS for security
    s.starttls()
    
    # Authentication
    s.login(fromaddr, base64.b64decode("c2ltcGxleDIwMjA=").decode("utf-8"))
    
    # Converts the Multipart msg into a string
    text = msg.as_string()
    
    # sending the mail
    s.sendmail(fromaddr, toaddr, text)
    
    # terminating the session
    s.quit()


def generate_report(report_data, template_file = "report_template.tex", to_email=None):
    instance_name = report_data['instance_name']
    
    output_path = reporting_path / 'temp/'
    reports_pdf_path = reporting_path / 'reports/'
    template_file = str(reporting_path / template_file)
    
    tex_file = str(output_path / f'report_{instance_name}.tex')
    
    pdf_file = output_path / f'report_{instance_name}.pdf'
    if not output_path.exists():
        os.system(f"mkdir {output_path}")
    if not reports_pdf_path.exists():
        os.system(f"mkdir {reports_pdf_path}")
    
    # Copy figures and rename paths
    os.system(f"scp {report_data['IHT_PLOT_FILE']} {reporting_path / 'fig1.pdf'}")
    os.system(f"scp {report_data['ToIHT_PLOT_FILE']} {reporting_path / 'fig2.pdf'}")
    report_data['IHT_PLOT_FILE'] = reporting_path / 'fig1.pdf'
    report_data['ToIHT_PLOT_FILE'] = reporting_path / 'fig2.pdf'
    
    fill_template(report_data, template_file, tex_file)
    os.system(f"/Library/TeX/texbin/pdflatex  -output-directory={output_path} {tex_file}")
    
    # Clean folder
    try:
        os.system(f"rm {reporting_path / 'fig1.pdf'}")
        os.system(f"rm {reporting_path / 'fig2.pdf'}")
    except Exception:
        pass
    
    # Send email if valid address provided
    if to_email and '@' in to_email:
        send_report(instance_name, to_email, pdf_file)
    
    os.system(f"scp {pdf_file} {reports_pdf_path}")
    
def generate_report_tier(report_data, template_file = "report_template_tier.tex", to_email=None):
    instance_name = report_data['instance_name']
    
    output_path = reporting_path / 'temp/'
    reports_pdf_path = reporting_path / 'reports/'
    template_file = str(reporting_path / template_file)
    
    tex_file = str(output_path / f'report_{instance_name}.tex')
    
    pdf_file = output_path / f'report_{instance_name}.pdf'
    if not output_path.exists():
        os.system(f"mkdir {output_path}")
    if not reports_pdf_path.exists():
        os.system(f"mkdir {reports_pdf_path}")
    
    # Copy figures and rename paths
    os.system(f"scp {report_data['IHT_PLOT_FILE']} {reporting_path / 'fig1.pdf'}")
    os.system(f"scp {report_data['ToIHT_PLOT_FILE']} {reporting_path / 'fig2.pdf'}")
    report_data['IHT_PLOT_FILE'] = reporting_path / 'fig1.pdf'
    report_data['ToIHT_PLOT_FILE'] = reporting_path / 'fig2.pdf'
    
    os.system(f"scp {report_data['ICU_PLOT_FILE']} {reporting_path / 'fig3.pdf'}")
    os.system(f"scp {report_data['ToICU_PLOT_FILE']} {reporting_path / 'fig4.pdf'}")
    report_data['ICU_PLOT_FILE'] = reporting_path / 'fig3.pdf'
    report_data['ToICU_PLOT_FILE'] = reporting_path / 'fig4.pdf'
    
    fill_template(report_data, template_file, tex_file)
    os.system(f"/Library/TeX/texbin/pdflatex  -output-directory={output_path} {tex_file}")
    
    # Clean folder
    try:
        os.system(f"rm {reporting_path / 'fig1.pdf'}")
        os.system(f"rm {reporting_path / 'fig2.pdf'}")
        os.system(f"rm {reporting_path / 'fig3.pdf'}")
        os.system(f"rm {reporting_path / 'fig4.pdf'}")
    except Exception:
        pass
    
    # Send email if valid address provided
    if to_email and '@' in to_email:
        send_report(instance_name, to_email, pdf_file)
    
    os.system(f"scp {pdf_file} {reports_pdf_path}")

