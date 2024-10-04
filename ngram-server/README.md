# VM Server Setup
## 1. Set ssh key
    Gen key: `ssh-keygen -t ed25519 -C <email>`

    Add public key to github: `~/.ssh/id_ed25519.pub`

    `eval "$(ssh-agent -s)"`

   `ssh-add ~/.ssh/id_ed25519`


## 2. Clone repository
    `git clone https://github.com/BjarniHaukur/ngram-llm-analysis`

## 3. Install rust

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

4. ## open firewall for http on all ports on [Google Cloud](https://console.cloud.google.com/net-security/firewall-manager/firewall-policies/details/default-allow-http?project=dd2430-llm-ngram)


# Start
```bash
cargo run
```