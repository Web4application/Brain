cd app/frontend
npm create vite@latest . --template react
npm install

docker-compose up --build
sudo apt update
sudo apt install certbot python3-certbot-nginx

sudo certbot --nginx -d braineuron.ai -d www.braineuron.ai

sudo systemctl restart nginx
sudo ufw allow 'Nginx Full'
sudo crontab -e
# Add this line:
0 0 * * * certbot renew --quiet

nano /etc/nginx/sites-available/braineuron.ai

ln -s /etc/nginx/sites-available/braineuron.ai /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

apt update && apt upgrade -y
apt install docker docker-compose nginx certbot python3-certbot-nginx git -y
ssh root@your_droplet_ip
