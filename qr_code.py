import qrcode
from qrcode.image.styledpil import StyledPilImage
from PIL import Image

def generate_qr_with_logo(url: str, logo_path: str, output_path: str = "qr.png"):
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High: 30% recovery
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    print("Overlaying logo...")
    logo = Image.open(logo_path).convert("RGBA")
    img_w, img_h = img.size
    logo_size = img_w // 4  # Logo takes up ~25% of QR width
    logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

    pos = ((img_w - logo_size) // 2, (img_h - logo_size) // 2)
    img.paste(logo, pos, mask=logo)
    img.save(output_path, dpi=(300, 300))
    print(f"Saved to {output_path}")


print("Generating QR code with logo...")
generate_qr_with_logo(
    url="https://tranquil-metatarsal-615.notion.site/CurriculuMoE-1890c4b9239d80ada024e952b89cdcac?pvs=143",
    logo_path="logo.png",
    output_path="qr_poster.png"
)