

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b a)
(on c e)
(ontable d)
(ontable e)
(ontable f)
(clear b)
(clear c)
(clear d)
)
(:goal
(and
(on a f)
(on b d)
(on c e)
(on d a)
(on e b))
)
)


