import torch


class EncoderDecoder(torch.nn.Module): 
    def __init__(self, in_channels=3): 
        super().__init__() 


        self.encoder = torch.nn.Sequential( 
            # input image: 200x200 
            # Block 1
            torch.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), # 200x200x16
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1), # 200x200x16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 100x100x16

            # Block 2 
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), # 100x100x32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 50x50x64

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), # 50x50x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 25x25x64

            # Block 3 
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), # 25x25x128
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), # 25x25x256
            torch.nn.ReLU(),
            
        )


        self.bottleneck = torch.nn.Sequential( 
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            torch.nn.ReLU()
        ) 

        self.decoder = torch.nn.Sequential( 
            # 25×25×512 → 50×50×256
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            
            # 50×50×256 → 100×100×128
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            
            # 100×100×128 → 200×200×64
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            
            #  200×200×64 → 200×200×1
            torch.nn.Conv2d(64, 1, kernel_size=1),

        )

    

    def forward(self, x): 
        x = self.encoder(x) 
        x = self.bottleneck(x) 
        x = self.decoder(x) 

        return x 